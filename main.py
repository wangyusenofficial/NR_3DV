import os
import time
import logging
import argparse
import cv2
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
import dataset.dataset_distribute
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
import torch.nn as nn
import time
import warnings
warnings.filterwarnings('ignore')


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # dataset
        self.dataset,self.dataloader = dataset.dataset_distribute.get_train_loader(dataset.dataset_distribute.distribute_dataset,self.conf['dataset'],self.conf['dataset']['worker'])
        self.dataset.batch_size = self.batch_size

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.nerf_outside = self.nerf_outside.float()
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.sdf_network = self.sdf_network.float()
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.deviation_network = self.deviation_network.float()
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        self.color_network = self.color_network.float()
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()


    def train(self):
        print(time.asctime())
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        # if self.end_epoch * self.dataset.n_images <= self.end_iter:
        #     self.end_iter = self.end_epoch * self.dataset.n_images
        # print(f'iter_end: {self.end_iter}')
        res_step = self.end_iter - self.iter_step
        for iter_i in tqdm(range(res_step)):
            if (iter_i+self.dataset.n_images) % self.dataset.n_images == 0:
                loader = iter(self.dataloader)
            data,alps = next(loader)
            data = data[0].cuda()
            alpha_quantile_point, alpha_mean = alps[0][0],alps[1]
            rays_o, rays_d, true_rgb, ray_len_mvs, ray_len_mvs_mask, near, far, surfacenorm, surfacealp, pert_rays_o, pert_rays_d = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], \
                  data[:, 10: 11], data[:, 11: 12], data[:, 12: 13], data[:, 13: 16], data[:,16:17], data[:,17:20], data[:,20:23]

            ray_norm_len = torch.norm(surfacenorm, dim=1)
            ray_norm_mask = surfacealp <= torch.max(alpha_quantile_point[7],alpha_mean[0])
            ray_norm = surfacenorm / (ray_norm_len + 1e-5).unsqueeze(-1)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())
            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']
            mid_z_vals = render_out['mid_z_vals']
            weights = render_out['weights']

            # Loss forward
            depth_estimate = torch.sum(weights[:, :mid_z_vals.shape[1]] * mid_z_vals, -1).reshape(-1, 1)
            depth_mvs_loss = self.smooth_l1_loss(depth_estimate * ray_len_mvs_mask, ray_len_mvs * ray_len_mvs_mask,
                                                 beta=0.2, reduction='sum') / (ray_len_mvs_mask.sum() + 1)

            in_samples = runner.renderer.n_samples + runner.renderer.n_importance
            est_normals = render_out['gradients'] * render_out['weights'][:, :in_samples, None]
            est_normals = est_normals.sum(dim=1)
            est_normals_len = torch.norm(est_normals, dim=1)
            est_normals = est_normals / (est_normals_len + 1e-5).unsqueeze(-1)
            norm_loss = self.smooth_l1_loss(est_normals * ray_norm_mask.unsqueeze(-1),
                                            ray_norm * ray_norm_mask.unsqueeze(-1), beta=0.2, reduction='sum',
                                            key=lambda x, y: (x - y).norm(dim=1)) / (
                                ray_norm_mask.sum() + 1)
            color_error = (color_fine - true_rgb)
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / self.batch_size
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2).sum() / (self.batch_size * 3.0)).sqrt())
            eikonal_loss = gradient_error

            depth_perturb_err = 0
            normal_perturb_err_gt = 0
            if self.dataset.smooth:
                render_out_res = self.renderer.render(pert_rays_o, pert_rays_d, near, far,
                                                  background_rgb=background_rgb,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio())
                color_fine_res = render_out_res['color_fine']
                s_val_res = render_out_res['s_val']
                cdf_fine_res = render_out_res['cdf_fine']
                gradient_error_res = render_out_res['gradient_error']
                weight_max_res = render_out_res['weight_max']
                weight_sum_res = render_out_res['weight_sum']
                mid_z_vals_res = render_out_res['mid_z_vals']
                weights_res = render_out_res['weights']

                # Loss smooth
                depth_estimate_res = torch.sum(weights_res[:, :mid_z_vals_res.shape[1]] * mid_z_vals_res, -1).reshape(
                    -1, 1)
                depth_perturb_err = self.smooth_l1_loss(depth_estimate_res, depth_estimate, reduction='sum',
                                                        beta=0.2) / self.batch_size

                in_samples = runner.renderer.n_samples + runner.renderer.n_importance
                est_normals_res = render_out_res['gradients'] * render_out_res['weights'][:, :in_samples, None]
                est_normals_res = est_normals_res.sum(dim=1)
                est_normals_len_res = torch.norm(est_normals_res, dim=1)
                est_normals_res = est_normals_res / (est_normals_len_res + 1e-5).unsqueeze(-1)

                normal_perturb_err_gt = self.smooth_l1_loss(est_normals_res * ray_norm_mask.unsqueeze(-1),
                                                            ray_norm * ray_norm_mask.unsqueeze(-1),
                                                            key=lambda x, y: (x - y).norm(dim=1), reduction='sum',
                                                            beta=0.2) / (ray_norm_mask.sum() + 1)
                normal_perturb_err_pre = self.smooth_l1_loss(est_normals_res, est_normals,
                                                             key=lambda x, y: (x - y).norm(dim=1), reduction='sum',
                                                             beta=0.2) / self.batch_size

            # overall loss
            loss = color_fine_loss + \
                   eikonal_loss * self.igr_weight + \
                   depth_mvs_loss + norm_loss * 0.001 + depth_perturb_err * 0.001 + normal_perturb_err_gt * 0.001
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/depth_perturb_err', depth_perturb_err, self.iter_step)
            self.writer.add_scalar('Loss/normal_perturb_err_gt', normal_perturb_err_gt, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Loss/normal_loss', norm_loss, self.iter_step)
            self.writer.add_scalar('Loss/depth_mvs_loss', depth_mvs_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1]).sum() / self.batch_size, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max).sum() / self.batch_size, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss,
                                                           self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

        print(time.asctime())

    def smooth_l1_loss(self, input, target, beta=1, reduction='none', key = None):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        if key != None:
            n = key(input,target)
        else:
            n = torch.abs(input - target)
        cond = n < beta
        ret = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        return ret

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, near, far = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        near = near.reshape(-1,1).split(self.batch_size)
        far = far.reshape(-1,1).split(self.batch_size)
        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch, near_batch, far_batch in zip(rays_o, rays_d, near, far):
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near_batch,
                                              far_batch,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_subset(self,img0_idx,img1_idx,resolution_level=1):
        self.batch_size=1024
        self.dataset.gen_subset(img0_idx,img1_idx,n_frames=5)
        for subidx in range(int(self.dataset.sub_fars.shape[0])):
            rays_o, rays_d, near, far = self.dataset.gen_rays_at_subset(subidx, resolution_level=resolution_level)
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
            near = near.reshape(-1,1).split(self.batch_size)
            far = far.reshape(-1,1).split(self.batch_size)
            out_rgb_fine = []

            for rays_o_batch, rays_d_batch, near_batch, far_batch in zip(rays_o, rays_d, near, far):
                background_rgb = None
                render_out = self.renderer.render(rays_o_batch,
                                                  rays_d_batch,
                                                  near_batch,
                                                  far_batch,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                  background_rgb=background_rgb)

                def feasible(key): return (key in render_out) and (render_out[key] is not None)

                if feasible('color_fine'):
                    out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                del render_out

            img_fine = None
            if len(out_rgb_fine) > 0:
                img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

            os.makedirs(os.path.join(self.base_exp_dir, 'rendering'), exist_ok=True)

            for i in range(img_fine.shape[-1]):
                if len(out_rgb_fine) > 0:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'rendering',
                                            'render_{}_{}_{}.png'.format(img0_idx, img1_idx, subidx)),
                               img_fine[..., i])


    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d, near, far = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        np.save(os.path.join(self.base_exp_dir,
                                'renderout',
                                'far_{}_{}_{}.npy'.format(idx_0, idx_1, 0)),
                   np.asarray(far))


        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        near = near.reshape(-1,1).split(self.batch_size)
        far = far.reshape(-1,1).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch, near_batch, far_batch in zip(rays_o, rays_d,near,far):
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch.cuda(),
                                              rays_d_batch.cuda(),
                                              near_batch.float().cuda(),
                                              far_batch.float().cuda(),
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        print(f'output dir {video_dir}')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()

    def interpolate_view_sep(self, img_idx_0, img_idx_1,n_frame,resolution_level=1,case='test'):
        images = []
        n_frames = n_frame
        print(f'interpolate {img_idx_0}, {img_idx_1}, n_frame{n_frame}, resolution {resolution_level}')
        os.makedirs(os.path.join(self.base_exp_dir,case,
                                    'renderout'), exist_ok=True)
        print(os.path.join(self.base_exp_dir,
                                    'renderout'))
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=resolution_level))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])
            cv.imwrite(os.path.join(self.base_exp_dir,case,
                                    'renderout',
                                    'renderout_{}_{}_{}.png'.format(img_idx_0, img_idx_1, i)),
                       images[i])


if __name__ == '__main__':

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')


    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
    elif args.mode == 'renderfull':
        for i in range(int(runner.dataset.n_images)-1):
            runner.render_subset(i,i+1,resolution_level=args.res)
