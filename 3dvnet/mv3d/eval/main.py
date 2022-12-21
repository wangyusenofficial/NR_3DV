from mv3d.dsets import dataset, scenelists, frameselector
import torch
import os
import numpy as np
from mv3d import config
from mv3d.eval import processresults
from mv3d.eval import config as eval_config
from mv3d.eval import meshtodepth
import open3d as o3d


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main(save_dirname, pred_func, dset_kwargs, net, overwrite=False, depth=True, scene='scene1111_11'):


    print(f'save_dirname {save_dirname}, scene {scene}')
    save_dir = os.path.join(eval_config.SAVE_DIR, save_dirname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print('Preparing dataset...')
    if eval_config.DATASET_TYPE == 'scannet':
        scenes = sorted(scenelists.get_scenes_scannet(config.SCANNET_DIR, 'test'))
    elif eval_config.DATASET_TYPE == 'scannet_val':
        scenes = sorted(scenelists.get_scenes_scannet(config.SCANNET_DIR, 'val'))
    elif eval_config.DATASET_TYPE == 'icl-nuim':
        scenes = sorted(scenelists.get_scenes_icl_nuim(config.ICL_NUIM_DIR))
    elif eval_config.DATASET_TYPE == 'tum-rgbd':
        scenes = sorted(scenelists.get_scenes_tum_rgbd(config.TUM_RGBD_DIR))
    elif scene != 'scene1111_11':
        scenes = [scene]
    else:
        raise NotImplementedError


    selector = frameselector.NextPoseDistSelector(eval_config.PDIST, 20)
    n_src_on_either_side = eval_config.N_SRC_ON_EITHER_SIDE
    
    dset = dataset.Dataset(scenes, selector, None, (480, 640), augment=False, n_src_on_either_side=n_src_on_either_side,
                           **dset_kwargs)

    net = net.to(DEVICE)
    net.eval()

    start_idx = 0
    for j, scene in enumerate(scenes[start_idx:]):
        scene_name = os.path.basename(scene)
        print(f'scene_name {scene_name}')
        print('{} / {}: {}'.format(j + 1 + start_idx, len(scenes), scene_name))
        scene_save_dir = os.path.join(save_dir, 'scenes', scene_name)
        if not os.path.exists(scene_save_dir):
            os.makedirs(scene_save_dir)
        pred_file_path = os.path.join(scene_save_dir, 'preds.npz')
        print(f'{pred_file_path}')

        # make predictions
        if not os.path.exists(pred_file_path) or overwrite:
            batch, _, _, img_idx = dset.get(j+start_idx, return_verbose=True, seed_idx=0)
            batch.__setattr__('images_batch', torch.zeros(batch.images.shape[0], dtype=torch.long))
            batch = batch.to(DEVICE)
            ref_idx = torch.unique(batch.ref_src_edges[0]).detach().cpu().numpy()

            if depth:
                depth_preds, init_prob, final_prob = pred_func(batch, scene, dset, net)
            else:
                mesh = pred_func(batch, scene, dset, net)
                o3d.io.write_triangle_mesh(os.path.join(scene_save_dir, 'mesh.ply'), mesh)

                # render depth predictions from mesh
                poses = torch.cat((batch.rotmats, batch.tvecs.unsqueeze(2)), dim=2).detach().cpu().numpy()
                eye_row = np.repeat(np.array([[[0., 0., 0., 1.]]]), poses.shape[0], axis=0)
                poses = np.concatenate((poses, eye_row), axis=1)
                K = batch.K.detach().cpu().numpy()

                depth_preds = meshtodepth.process_scene(mesh, poses[ref_idx], K[ref_idx])
                init_prob = None
                final_prob = None

            # convert K to size of depth images
            old_h, old_w = batch.images.shape[-2:]
            new_h, new_w = depth_preds.shape[-2:]
            x_fact = float(new_w) / float(old_w)
            y_fact = float(new_h) / float(old_h)
            K = batch.K[ref_idx]
            K[:, 0, :] *= x_fact
            K[:, 1, :] *= y_fact

            # save depth prediction results to preds.npz file, to be used for calculating 2D/3D metrics
            preds = dict(
                scene=os.path.basename(scene),
                depth_preds=depth_preds,
                rotmats=batch.rotmats[ref_idx].detach().cpu().numpy(),
                tvecs=batch.tvecs[ref_idx].detach().cpu().numpy(),
                K=K.detach().cpu().numpy(),
                img_idx=img_idx[ref_idx],       # stores indices of all reference images
            )
            # save probability maps for fmvs and pmvs to be used in point cloud fusion
            if init_prob is not None:
                preds['init_prob'] = init_prob
            if final_prob is not None:
                preds['final_prob'] = final_prob

            np.savez(
                pred_file_path,
                **preds
            )

        # calculated 2D and 3D metrics from the network predictions in preds.npz file
        # processresults.process_scene_2d_metrics(scene, scene_save_dir, overwrite)
        torch.cuda.empty_cache()
        # if depth:
        #     processresults.process_depth_3d_metrics(
        #         scene, scene_save_dir, eval_config.Z_THRESH, eval_config.RUN_TSDF_FUSION, eval_config.RUN_PCFUSION,
        #         overwrite)
        # else:
        #     processresults.process_volume_3d_metrics(scene, scene_save_dir, overwrite)

    # processresults.calc_avg_metrics(save_dir)   # calculate and save aggregated metrics


# Traceback (most recent call last):
#   File "mv3d/eval-3dvnet.py", line 143, in <module>
#     main('3dvnet', process_scene, dset_kwargs, net, depth=True,scene=args.scene)
#   File "/mnt/data2/yswang_data2/code/distribute_neuralroom_3dv/3dvnet/mv3d/eval/main.py", line 114, in main
#     processresults.process_depth_3d_metrics(
#   File "/mnt/data2/yswang_data2/code/distribute_neuralroom_3dv/3dvnet/mv3d/eval/processresults.py", line 265, in process_depth_3d_metrics
#     depth_gt_reproj = meshtodepth.process_scene(gt_mesh, poses, K)
#   File "/mnt/data2/yswang_data2/code/distribute_neuralroom_3dv/3dvnet/mv3d/eval/meshtodepth.py", line 52, in process_scene
#     renderer = Renderer(mesh, *render_size)
#   File "/mnt/data2/yswang_data2/code/distribute_neuralroom_3dv/3dvnet/mv3d/eval/meshtodepth.py", line 13, in __init__
#     self.renderer = pyrender.OffscreenRenderer(width, height)
#   File "/home/yswang/anaconda3/envs/nr3dv/lib/python3.8/site-packages/pyrender/offscreen.py", line 31, in __init__
#     self._create()
#   File "/home/yswang/anaconda3/envs/nr3dv/lib/python3.8/site-packages/pyrender/offscreen.py", line 149, in _create
#     self._platform.init_context()
#   File "/home/yswang/anaconda3/envs/nr3dv/lib/python3.8/site-packages/pyrender/platforms/pyglet_platform.py", line 50, in init_context
#     self._window = pyglet.window.Window(config=conf, visible=False,
#   File "/home/yswang/anaconda3/envs/nr3dv/lib/python3.8/site-packages/pyglet/window/xlib/__init__.py", line 168, in __init__
#     super(XlibWindow, self).__init__(*args, **kwargs)
#   File "/home/yswang/anaconda3/envs/nr3dv/lib/python3.8/site-packages/pyglet/window/__init__.py", line 548, in __init__
#     display = pyglet.canvas.get_display()
#   File "/home/yswang/anaconda3/envs/nr3dv/lib/python3.8/site-packages/pyglet/canvas/__init__.py", line 94, in get_display
#     return Display()
#   File "/home/yswang/anaconda3/envs/nr3dv/lib/python3.8/site-packages/pyglet/canvas/xlib.py", line 123, in __init__
#     raise NoSuchDisplayException('Cannot connect to "%s"' % name)
# pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"

