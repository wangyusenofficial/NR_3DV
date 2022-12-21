import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os

def load_depth(filename,depth_shift=1):
    '''

    :param filename: depth file, h,w
    :param depth_shift:
    :return: depth * depth shift,   h,w,1 numpy matrix
    '''
    depth = cv2.imread(filename,-1) * depth_shift
    h,w = depth.shape
    return depth.reshape(h,w,1)

def load_norm(filename):
    '''
    return a h,w,3 numpy normal map
    :param filename: pfm file
    :return: h,w,3 numpy normal map
    '''
    return cv2.imread(filename, -1)

def load_alpha(filename):
    '''

    :param filename: pfm file
    :return: h,w,1 numpy alpha map
    '''
    alpha = cv2.imread(filename, -1)
    h,w = alpha.shape
    return alpha.reshape(h,w,1)

def load_color(filename):
    '''
    load image file, return a h,w,3 numpy matrix
    :param filename:
    :return:
    '''
    images_np = cv2.imread(filename) / 256.0
    return images_np

def load_far(filename):
    far = cv2.imread(filename,-1)
    h,w = far.shape
    return far.reshape(h,w,1)

def load_cam(filename):
    return np.load(filename)

def transform_norm(norm, extrin):
    norm_gt = norm.copy()
    extrin_inv = np.linalg.pinv(extrin)
    R = extrin_inv[:3, :3]
    h, w, c = norm_gt.shape
    norm_gt = norm_gt.reshape(-1, 3)
    norm_gt = np.matmul(R, norm_gt.transpose())
    norm_gt = norm_gt.transpose()
    norm_gt = norm_gt.reshape(h, w, c)
    return norm_gt

# def get_normal_prior(normal_root, poses, save_path):
#     img_list = [i.split('.')[0] for i in colorlist]
#     normlist = []
#     for idx in range(len(img_list)):
#         normname = f'{img_list[idx]}.npy'
#         normmap = np.load(os.path.join(normal_root, normname))[0] * -1
#         normlist.append(transform_norm(normmap, np.linalg.pinv(poses[idx])))
#     np.save(os.path.join(save_path, 'surface_normal.npy'), normlist)

def preproc_normal(normal_root, dv3_recfile, output, sortfunc=None,index=True):
    '''
    prepare normal and alpha files from surface normal uncertainty output, the filename should not contain '.' ! the output normal is a pfm file contains a h,w,3 matrix, alpha is a pfm file contains a h,w matrix
    :param normal_root: normal uncertainty output root
    :param dv3_recfile: 3dvnet output preds.npz
    :param output: output directory
    :param sort: sort function for a list
    :param index: if output name use idx, or use original name
    :return: None
    '''
    print(f'processing {normal_root}')
    os.makedirs(os.path.join(output,'normal'),exist_ok=True)
    os.makedirs(os.path.join(output,'alpha'),exist_ok=True)
    nalist = glob.glob(os.path.join(normal_root,'*.npy'))
    rec_file = np.load(dv3_recfile)

    if sortfunc is None:
        nalist.sort()
    else:
        nalist.sort(key=sortfunc)
    nfile = len(nalist) // 2
    assert len(nalist) % 2 == 0, 'alpha and normal do not match'
    assert nfile == rec_file['img_idx'].__len__(), '3dv recfile does not match the normal file'
    poses = []
    extrins = []
    for i in range(len(nalist)//2):
        extrin = np.eye(4)
        R = rec_file['rotmats'][i]
        t = rec_file['tvecs'][i]
        extrin[:3, :3] = R
        extrin[:3, 3] = t
        extrins.append(extrin)
        poses.append(np.linalg.pinv(extrin))
    extrins = np.stack(extrins)
    poses = np.stack(poses)


    for idx in range(nfile):
        alpha = np.load(nalist[idx*2])
        norm = np.load(nalist[idx*2+1])[0] * -1
        norm = transform_norm(norm, extrins[idx])
        fname = nalist[idx].split('.')[0].split('/')[-1]
        if index:
            fname = idx.__str__().zfill(6)
        cv2.imwrite(os.path.join(output, 'normal', f'{fname}.pfm'), norm)
        cv2.imwrite(os.path.join(output, 'alpha', f'{fname}.pfm'), alpha[0])

def distribute_all(file, output, mode):
    '''

    :param file: n,h,w,c
    :param output: split h,w (if c==1)  or h,w,c (if c != 1)
    :param mode: depth/far/normal/alpha
    :return:
    '''
    integ = np.load(file)
    output = os.path.join(output,mode)
    os.makedirs(output,exist_ok=True)
    idx = 0
    for i in range(integ.shape[0]):
        cv2.imwrite(os.path.join(output,f'{idx.__str__().zfill(6)}.pfm'),integ[idx])
        idx += 1


if __name__ == '__main__':

    file = '/home/yswang/server160data2/dsb_0785/fake_neus_input/surface_far.npy'
    mode = 'far'
    output = '/home/yswang/server160data2/dsb_0785/fake_neus_input'
    distribute_all(file,output,'far')

    file = '/home/yswang/server160data2/dsb_0785/fake_neus_input/depth.npy'
    mode = 'depth'
    output = '/home/yswang/server160data2/dsb_0785/fake_neus_input'
    distribute_all(file,output,mode)

    file = '/home/yswang/server160data2/dsb_0785/fake_neus_input/surface_alpha.npy'
    mode = 'alpha'
    output = '/home/yswang/server160data2/dsb_0785/fake_neus_input'
    distribute_all(file,output,mode)

    file = '/home/yswang/server160data2/dsb_0785/fake_neus_input/surface_normal.npy'
    mode = 'normal'
    output = '/home/yswang/server160data2/dsb_0785/fake_neus_input'
    distribute_all(file,output,mode)
    pass

