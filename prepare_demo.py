import os
import prior_utils.utils as utils
import argparse
import numpy as np
import cv2
import glob
import open3d as o3d
from dataset.dataset_distribute import distribute_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/yswang/data2/scans')
    args = parser.parse_args()
    root = args.root

    os.makedirs(os.path.join(root,'normal'),exist_ok=True)
    os.makedirs(os.path.join(root,'alpha'),exist_ok=True)
    os.makedirs(os.path.join(root,'far'),exist_ok=True)

    cams = np.load(os.path.join(root,'cameras.npz'))
    poses=[]
    for i in range(len(cams.files)):
        checkname = cams.files[i]
        if checkname.split('_')[0] == 'world':
            intrinsic, pose = distribute_dataset.load_K_Rt_from_P(P=cams[checkname][:3,:])
            poses.append(pose)

    # prepare normal
    print('generate normal prior')
    os.system(f'cp {os.path.join(root, "image/*")} surface_normal_uncertainty/examples/')
    os.system(f'cd surface_normal_uncertainty/ ; python test.py --pretrained scannet --architecture BN')

    nalist = glob.glob('surface_normal_uncertainty/examples/results/*.npy')
    nalist.sort()
    nfile = len(nalist) // 2
    assert len(nalist) % 2 == 0, 'alphas and normals do not match'
    assert nfile == len(poses), 'cams and normals do not match'

    for idx in range(nfile):
        alpha = np.load(nalist[idx*2])
        norm = np.load(nalist[idx*2+1])[0] * -1
        norm = utils.transform_norm(norm, np.linalg.pinv(poses[idx]))
        fname = nalist[idx].split('.')[0].split('/')[-1]
        cv2.imwrite(os.path.join(root, 'normal', f'{fname}.pfm'), norm)
        cv2.imwrite(os.path.join(root, 'alpha', f'{fname}.pfm'), alpha[0])

    os.system(f'rm surface_normal_uncertainty/examples/*.png')
    os.system(f'rm -rf surface_normal_uncertainty/examples/results')

    # generate ray len
    print('generate ray length')
    downscale=2
    os.listdir(os.path.join(root,'image'))
    h,w,_ = cv2.imread(os.path.join(root,'image',os.listdir(os.path.join(root,'image'))[0])).shape
    intrinsic[:2,:] = intrinsic[:2,:]/downscale
    d_h,d_w = int(h/2),int(w/2)
    cube = o3d.io.read_triangle_mesh(os.path.join(root,'boundingbox.ply'))
    ray_len = utils.get_ray_tracing_init(np.asarray(poses),intrinsic[:3,:3],cube,h=d_h,w=d_w,mesh_f_name_flag=False, depth_flag=False)

    for i in range(ray_len.shape[0]):
        cv2.imwrite(os.path.join(root, 'far', f'{i.__str__().zfill(6)}.pfm'), ray_len[i])