import os
import matplotlib.pyplot as plt
from pyhocon import ConfigFactory
import torch.nn.functional as F
import cv2
import numpy as np
import torch.utils.data as data
import torch
import dataset.data_utils
import glob

class distribute_dataset(data.Dataset):
    def __init__(self,conf,mode='train',batch_size=512):
        super().__init__()
        print('using distribute dataset')
        self.conf = conf
        self.mode = mode
        self.batch_size = batch_size
        self.n_images = len(os.listdir(os.path.join(self.conf['data_dir'],'image')))
        self.smooth = self.conf['smooth']
        self.color_list = glob.glob(os.path.join(self.conf['data_dir'],'image','*'))
        self.color_list.sort()
        self.normal_list = glob.glob(os.path.join(self.conf['data_dir'],'normal','*'))
        self.normal_list.sort()
        self.alpha_list = glob.glob(os.path.join(self.conf['data_dir'],'alpha','*'))
        self.alpha_list.sort()
        self.depth_list = glob.glob(os.path.join(self.conf['data_dir'],'depth','*'))
        self.depth_list.sort()
        self.far_list = glob.glob(os.path.join(self.conf['data_dir'],'far','*'))
        self.far_list.sort()

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')
        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict


        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a boundingbox at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.scale = torch.from_numpy(np.asarray([scale_mat[0, 0] for scale_mat in self.scale_mats_np]))

        self.intrinsics_all = []
        self.pose_all_raw = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = self.load_K_Rt_from_P(P)
            # pose : cam to world matrix
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

            intrinsics, pose = self.load_K_Rt_from_P(world_mat[:3, :4])
            self.pose_all_raw.append(torch.from_numpy(pose).float())
        self.pose_all_raw = torch.stack(self.pose_all_raw)

        self.intrinsics_all = torch.stack(self.intrinsics_all)  # [n_images, 4, 4]    .to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all)  # [n_images, 4, 4] .to(self.device)
        self.H, self.W, _ = cv2.imread(self.color_list[0]).shape
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        print(object_scale_mat)
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

    @staticmethod
    def load_K_Rt_from_P(P):
        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]
        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        return intrinsics, pose

    def __len__(self):
        return self.n_images

    def __getitem__(self, i):
        torch.set_default_tensor_type('torch.FloatTensor')
        idx=i
        color = torch.from_numpy(dataset.data_utils.load_color(self.color_list[idx])).float().unsqueeze(dim=0) #1hw3, has normalized to 0-1
        normal = torch.from_numpy(dataset.data_utils.load_norm(self.normal_list[idx])).float().unsqueeze(dim=0) #1hw3
        depth = torch.from_numpy(dataset.data_utils.load_depth(self.depth_list[idx])).float().unsqueeze(dim=0) #1hw1
        alpha = torch.from_numpy(dataset.data_utils.load_alpha(self.alpha_list[idx])).float().unsqueeze(dim=0) #1hw1
        far = torch.from_numpy(dataset.data_utils.load_far(self.far_list[idx])).float().unsqueeze(dim=0) #1hw1

        if normal.shape[1] != self.H or normal.shape[2] != self.W:
            normal = F.interpolate(normal.permute(0, 3, 1, 2), size=(self.H, self.W), mode='bilinear').permute(0, 2, 3, 1)
            alpha = F.interpolate(alpha.permute(0, 3, 1, 2), size=(self.H, self.W), mode='bilinear').permute(0, 2, 3, 1)
        if depth.shape[1] != self.H or depth.shape[2] != self.W:
            depth = F.interpolate(depth.permute(0, 3, 1, 2), size=(self.H, self.W), mode='nearest').permute(0, 2, 3, 1)
        if far.shape[1] != self.H or far.shape[2] != self.W:
            far = F.interpolate(far.permute(0, 3, 1, 2), size=(self.H, self.W), mode='nearest').permute(0, 2, 3, 1)

        pixels_x = torch.randint(low=0, high=self.W, size=[self.batch_size]).cpu()
        pixels_y = torch.randint(low=0, high=self.H, size=[self.batch_size]).cpu()
        data = self.gen_random_rays_at(color,depth,normal,alpha,far,pixels_x,pixels_y,idx)
        return data

    def get_raw_item(self,idx):
        raw_dic={
        'color': torch.from_numpy(dataset.data_utils.load_color(self.color_list[idx])).float().unsqueeze(dim=0), #1hw3, has normalized to 0-1
        'normal': torch.from_numpy(dataset.data_utils.load_norm(self.normal_list[idx])).float().unsqueeze(dim=0), #1hw3
        'depth': torch.from_numpy(dataset.data_utils.load_depth(self.depth_list[idx])).float().unsqueeze(dim=0), #1hw1
        'alpha': torch.from_numpy(dataset.data_utils.load_alpha(self.alpha_list[idx])).float().unsqueeze(dim=0), #1hw1
        'far': torch.from_numpy(dataset.data_utils.load_far(self.far_list[idx])).float().unsqueeze(dim=0), #1hw1
        'intrin': self.intrinsics_all_inv[idx],
        'pose': self.pose_all[idx]}
        return raw_dic

    def gen_random_rays_at(self,color,depth,normal,alpha,far,pixels_x,pixels_y,img_idx):
        '''
        return data on cpu
        :param color:
        :param depth:
        :param normal:
        :param alpha:
        :param far:
        :param pixels_x:
        :param pixels_y:
        :param img_idx:
        :return:
        '''
        ray_color = color[0][(pixels_y, pixels_x)]  # batch_size, 3
        ray_norm = normal[0][(pixels_y, pixels_x)]  # batch_size, 3
        ray_alpha = alpha[0][(pixels_y, pixels_x)]  # batch_size, 1
        ray_depth = depth[0][(pixels_y, pixels_x)]  # batch_size, 1
        scale = self.scale[0]
        alpha_quantile_point = torch.quantile(ray_alpha, torch.tensor((0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)))
        alpha_mean = torch.mean(ray_alpha)
        ray_near = torch.ones_like(ray_alpha)*0.2/scale
        ray_far = far[0][(pixels_y, pixels_x)]/scale

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)],dim=-1).float()  # batch_size, 3
        point_mvs = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()
        point_mvs = point_mvs * ray_depth
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        point_mvs = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], point_mvs[:, :, None]).squeeze()
        ray_len_mvs = torch.linalg.norm(point_mvs, ord=2, dim=-1, keepdim=True) / scale
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        ray_mask_mvs = torch.logical_and(torch.logical_and(ray_len_mvs > 0,ray_len_mvs>= ray_near),ray_len_mvs<= ray_far)

        if self.smooth:
            perturb_x = torch.rand([self.batch_size]).cpu() * 1.2 - 0.6 + pixels_x.detach_()
            perturb_y = torch.rand([self.batch_size]).cpu() * 1.2 - 0.6 + pixels_y.detach_()
            p_pert = torch.stack([perturb_x, perturb_y, torch.ones_like(perturb_y)], dim=-1).float()  # batch_size, 3
            p_pert = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p_pert[:, :, None]).squeeze()  # batch_size, 3
            rays_v_pert = p / torch.linalg.norm(p_pert, ord=2, dim=-1, keepdim=True)  # batch_size, 3
            rays_v_pert = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v_pert[:, :, None]).squeeze()  # batch_size, 3
            rays_o_pert = self.pose_all[img_idx, None, :3, 3].expand(rays_v_pert.shape)  # batch_size, 3
        else:
            rays_o_pert = torch.zeros_like(rays_o)
            rays_v_pert = torch.zeros_like(rays_v)
        return torch.cat([rays_o.cpu(), rays_v.cpu(), ray_color,
                          ray_len_mvs.cpu(), ray_mask_mvs.reshape(self.batch_size, 1),
                          ray_near.cpu(), ray_far.cpu(),
                          ray_norm.cpu(),
                          ray_alpha,rays_o_pert.cpu(),rays_v_pert.cpu()
                          ], dim=-1),(alpha_quantile_point,alpha_mean)  # 3+3+3+1+1+1+1+3+1+3+3

    def gen_rays_at(self, idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        print(f'gen rays at -- H: {self.H}, W: {self.W}')
        far = torch.from_numpy(dataset.data_utils.load_far(self.far_list[idx])).float().unsqueeze(dim=0) #1hw1
        if far.shape[1] != self.H or far.shape[2] != self.W:
            far = F.interpolate(far.permute(0, 3, 1, 2), size=(self.H, self.W), mode='nearest').permute(0, 2, 3, 1)

        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l).cpu()
        ty = torch.linspace(0, self.H - 1, self.H // l).cpu()
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        scale = self.scale[0]

        ray_far = far[0][pixels_y.type(torch.long), pixels_x.type(torch.long)]/scale
        ray_near = torch.ones_like(ray_far) * 0.2 / scale

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1).cuda(), rays_v.transpose(0, 1).cuda(), ray_near.transpose(0, 1).cuda(), ray_far.transpose(0, 1).cuda()

    def image_at(self, idx, resolution_level):
        """
        clip 3 color channels to (0-255) & resize the image
        """
        img = cv2.imread(self.color_list[idx])
        return (cv2.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


def get_train_loader(dataset_func, conf, num_workers=8):
    dataset = dataset_func(conf)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers,pin_memory=False,
                             drop_last=False,)
    return dataset,loader

if __name__ == '__main__':

    data_conf_path = '/home/yswang/Downloads/gitcode/neus1214/distribute_dataloader/dataset.conf'
    f = open(data_conf_path)
    conf_text = f.read()
    f.close()
    conf = ConfigFactory.parse_string(conf_text)

    dataset,loader = get_train_loader(distribute_dataset,conf['dataset'],4)
    print(dataset.object_bbox_min)
    print(dataset.object_bbox_max)


    # idx = 298
    # d = dataset.get_raw_item(idx)
    #
    # rec = np.load('/home/yswang/server160/yswang_code/3dvnet/scannet_output/3dvnet/scenes/scene0713_00/preds.npz') #['scene', 'depth_preds', 'rotmats', 'tvecs', 'K', 'img_idx']
    # trueidx = rec['img_idx'][idx]
    #
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(d['depth'][0,:,:,0])
    # plt.subplot(2,2,2)
    # plt.imshow(d['color'][0])
    # plt.subplot(2,2,3)
    # plt.imshow(d['normal'][0]*0.5+0.5)
    # plt.subplot(2,2,4)
    # plt.imshow(rec['depth_preds'][idx])
    #
    # rec['K'][idx]
    # d['intrin']
    #
    #
    # iters = 200
    # print(loader.dataset.__len__())
    # epoch = np.ceil(iters / loader.dataset.n_images)
    # for i in range(int(epoch)):
    #     for sample in loader:
    #         sample[0].cuda()

            # print(f'i {sample["i"]}, idx {sample["idx"]}')


