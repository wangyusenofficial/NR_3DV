import numpy as np
import open3d as o3d
import os
import cv2
import glob
import requests


def alter(file1,file2,old_str,new_str):
  with open(file1, "r", encoding="utf-8") as f1,open(file2, "w", encoding="utf-8") as f2:
     for HH in f1:
       if old_str in HH:
         HH = HH.replace(old_str, new_str)
       f2.write(HH)


def tsdf_fusion(preds_root, data_root,voxel_size=4,save_root='default'):
    # dv3root = '/home/yswang/server160/yswang_code/3dvnet/scannet_output/3dvnet/scenes/scene0744_00'
    # color_root = '/home/yswang/server160data2/dv3net_data/scans_test/scene0744_00/color'

    color_root = os.path.join(data_root,'color')
    colorlist = os.listdir(color_root)
    colorlist.sort(key=lambda x:int(x.split('.')[0]))
    rec_file = np.load(os.path.join(preds_root,'preds.npz'))
    # print(f'processing {rec_file["scene"]}')
    img_idx = rec_file['img_idx']
    h, w = rec_file['depth_preds'][0].shape
    intrin = rec_file['K'][0]

    poses = []
    for i in range(len(img_idx)):
        extrin = np.eye(4)
        R = rec_file['rotmats'][i]
        t = rec_file['tvecs'][i]
        extrin[:3, :3] = R
        extrin[:3, 3] = t
        poses.append(np.linalg.pinv(extrin))
    poses = np.stack(poses)

    colors = []
    for i in img_idx:
        color = cv2.imread(os.path.join(color_root, colorlist[i]))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = cv2.resize(color,(w,h))
        colors.append(color)

    cam_intr = intrin[:3, :3]
    voxel_size = voxel_size

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size) / 200,
        sdf_trunc= 6 * float(voxel_size) / 200,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8) # open3d.cuda.pybind.pipelines.integration.TSDFVolumeColorType

    for i in range(poses.shape[0]):
        cam_pose = poses[i]
        depthmap = np.asarray(rec_file['depth_preds'][i]*1000,np.uint16)
        depthmap[depthmap>=4.5*1000] = 0
        depth_rgbd = o3d.geometry.Image(depthmap)
        # color_im = np.repeat(depthmap[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8)
        # color_im = np.repeat(depthmap[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8)
        # color_rgbd = o3d.geometry.Image(color_im)
        color = colors[i]
        color_rgbd = o3d.geometry.Image(color)
        # color_rgbd = o3d.geometry.Image(color_im)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_rgbd, depth_rgbd,convert_rgb_to_intensity=False,depth_trunc=8)
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=cam_intr[0, 0], fy=cam_intr[1, 1],
                                              cx=cam_intr[0, 2],
                                              cy=cam_intr[1, 2]), np.linalg.inv(cam_pose))
    e = volume.extract_triangle_mesh()
    if save_root == 'default':
        save_root = preds_root
    o3d.io.write_triangle_mesh(os.path.join(save_root,f'tsdf_voxel{voxel_size}.ply'), e)


def mask_black_side(img, thred=10):
    h,w,c=img.shape
    binary_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            if binary_image[i,j] <= thred:
                mask[i,j] = 1
            else:
                break
    for i in range(w):
        for j in range(h):
            if binary_image[j,i] <= thred:
                mask[j,i] = 1
            else:
                break
    for i in range(h):
        for j in range(w-1,-1,-1):
            if binary_image[i,j] <= thred:
                mask[i,j] = 1
            else:
                break
    for i in range(w):
        for j in range(h-1,-1,-1):
            if binary_image[j,i] <= thred:
                mask[j,i] = 1
            else:
                break
    return mask


def get_intrinsic(intrin_file):
    intrin = []
    with open(os.path.join(intrin_file), 'r') as f:
        for line in f:
            intrin += [float(value) for value in line.strip().split(' ') if value.strip() != '']
            continue
    intrin = np.array(intrin).reshape((4, 4))
    return intrin


def get_bbox_from_pcd(pcd, clip=False, min_len=3,scale=1.2):
    # obb = pcd.get_oriented_bounding_box()
    # obb.color = (0, 1, 0)
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    # o3d.visualization.draw_geometries([pcd, aabb])
    obb = aabb.get_oriented_bounding_box()
    # obb.color = (0, 1, 0)
    if clip:
        obb.extent = obb.extent * scale
        obb.extent = obb.extent.clip(min=min_len)
    aabb2 = obb.get_axis_aligned_bounding_box()
    aabb2.color = (1, 0, 0)
    # o3d.visualization.draw_geometries([pcd, aabb2])
    cube = o3d.geometry.TriangleMesh.create_box(width=aabb2.get_extent()[0],height=aabb2.get_extent()[1],depth=aabb2.get_extent()[2])
    cube.translate(
        aabb2.get_center(),
        relative=False,
    )
    # o3d.visualization.draw_geometries([cube, aabb2])
    return cube, aabb2


def write_cam_npz(normalization_mat, intrin, poses, save_path, cameras_filename):
    '''
    :param cam_folder: which contains split cam file
    :param normalization_mat:
    :param cam_num: int
    :param save_path:
    :param cameras_filename: save name
    :return:
    '''
    cameras_new = {}
    new_idx = 0
    for idx in range(len(poses)):
        intrin = intrin[:3, :3]
        extrin = np.linalg.pinv(poses[idx])[:3, :]
        proj_mat = intrin @ extrin
        cameras_new['scale_mat_%d' % new_idx] = normalization_mat
        cameras_new['world_mat_%d' % new_idx] = np.concatenate((proj_mat, np.array([[0, 0, 0, 1.0]])),
                                                               axis=0).astype(np.float32)
        new_idx += 1
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.savez('{0}/{1}.npz'.format(save_path, cameras_filename), **cameras_new)


def get_bbox1():
    p1 = np.asarray([1,1,1])
    p2 = np.asarray([-1,1,1])
    p3 = np.asarray([1,-1,1])
    p4 = np.asarray([1,1,-1])
    p5 = np.asarray([-1,-1,1])
    p6 = np.asarray([1,-1,-1])
    p7 = np.asarray([-1,1,-1])
    p8 = np.asarray([-1,-1,-1])
    bbox_point = np.stack([p1,p2,p3,p4,p5,p6,p7,p8])
    #
    #
    bbox = o3d.geometry.LineSet()
    bbox_line_idx = []
    for i in range(8):
        for j in range(i+1,8):
            bbox_line_idx.append([i,j])
    bbox_line_idx = np.asarray(bbox_line_idx)
    bbox.points = o3d.utility.Vector3dVector(bbox_point)
    bbox.lines = o3d.utility.Vector2iVector(bbox_line_idx)
    bbox.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([[255,0,0]]),(bbox_line_idx.shape[0],1)))
    return bbox


class GenRay():
    def __init__(self, H, W, K):
        u, v = np.meshgrid(np.arange(0, W), np.arange(0, H))
        p = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)
        p = np.matmul(np.linalg.pinv(K), p.transpose())
        self.rays_v = p / np.linalg.norm(p, ord=2, axis=0, keepdims=True)

    def gen_rays_at(self,pose):
        """
        Generate rays at world space from one camera.
        pose, not extrinsic!!!!!
        """
        rays_v = np.matmul(pose[:3, :3], self.rays_v).transpose()
        rays_o = np.tile(pose[:3, 3].reshape(1,3),(rays_v.shape[0],1))
        return np.concatenate((rays_o,rays_v),axis=1)  # [rays,6]

    def get_principal_v(self,K,pose):
        u,v =K[0,2],K[1,2]
        p = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)
        p = np.matmul(np.linalg.pinv(K), p.transpose())
        rays_v = p / np.linalg.norm(p, ord=2, axis=0, keepdims=True)
        rays_v = np.matmul(pose[:3, :3], rays_v).transpose()
        rays_o = np.tile(pose[:3, 3].reshape(1,3),(rays_v.shape[0],1))
        return np.concatenate((rays_o,rays_v),axis=1)


def get_ray_tracing_init(poses, intrin, mesh_f, bound_mesh=None, h=968, w=1296, mesh_f_name_flag=True, depth_flag=False, return_hit_id=False):
    '''

    :param poses: [237, 4, 4] numpy
    :param intrin: [3,3] numpy
    :param mesh_f: mesh name
    :param h:
    :return: surface_ray_len [nhw1] intersection of the surface
    '''
    print('generate ray_len from bounding box (aabb)')
    n_image = poses.shape[0]
    if mesh_f_name_flag:
        mesh = o3d.io.read_triangle_mesh(mesh_f)
    else:
        mesh = mesh_f
    # mesh_gt.compute_vertex_normals()
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    mesh_id_object = scene.add_triangles(mesh)
    # print(f'mesh_f id is : {mesh_id_object}')
    if bound_mesh is not None:
        bound_mesh = o3d.t.geometry.TriangleMesh.from_legacy(bound_mesh)
        mesh_id_bound = scene.add_triangles(bound_mesh)
        # print(f'bound_mesh id is : {mesh_id_bound}')
    surface_ray_len = np.zeros((n_image, h, w, 1))
    hit_id = np.zeros((n_image, h, w, 1))
    for i in range(poses.__len__()):
        if i%25 == 0:
            print(f'processing {i}/{poses.__len__()}')
        pose = poses[i]
        H = h
        W = w
        GR = GenRay(H=H, W=W, K=intrin[:3, :3])
        original_ray=GR.get_principal_v(K=intrin[:3, :3],pose=pose)

        rays = GR.gen_rays_at(pose)
        rays = rays.astype(np.float32)
        rays_v = rays[:,3:6]

        adj_cos = np.dot(rays_v,original_ray[0,3:6])
        rays_o3d = o3d.core.Tensor(rays,
                               dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays_o3d)
        t_img = ans['t_hit'].numpy()
        t_idx = ans['geometry_ids'].numpy().reshape(H,W)
        # print(t_idx)
        if depth_flag:
            ray_len = (t_img * adj_cos).reshape(H, W)
        else:
            ray_len = (t_img).reshape(H, W)
        surface_ray_len[i, :, :, 0] = ray_len
        hit_id[i,:,:,0] = t_idx
    if return_hit_id:
        return surface_ray_len, hit_id
    else:
        return surface_ray_len


def prepare_data_3dv(preds_root, rawdata_root,save_root='default',depthtrim=4.5):
    if save_root=='default':
        save_root = preds_root
    os.makedirs(os.path.join(save_root), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'image'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'alpha'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'far'), exist_ok=True)
    # meta info
    color_root = os.path.join(rawdata_root,'color')
    colorlist = os.listdir(color_root)
    colorlist.sort(key=lambda x:int(x.split('.')[0]))
    rec_file = np.load(os.path.join(preds_root,'preds.npz'))
    # print(f'processing {rec_file["scene"]}')
    img_idx = rec_file['img_idx']
    d_h, d_w = rec_file['depth_preds'][0].shape

    idx = 0
    # generate image and depth
    for i in range(len(img_idx)):
        img = cv2.imread(os.path.join(rawdata_root,'color',colorlist[img_idx[i]]),-1)
        mask = mask_black_side(cv2.resize(img,(d_w,d_h),interpolation=cv2.INTER_NEAREST))
        mask = np.logical_not(np.asarray(cv2.dilate(mask,(5,5),1),dtype=np.bool))
        depth = rec_file['depth_preds'][i]*mask
        depth[depth>=depthtrim]=0
        cv2.imwrite(os.path.join(save_root, 'image',f'{idx.__str__().zfill(6)}.png'),img)
        cv2.imwrite(os.path.join(save_root, 'depth',f'{idx.__str__().zfill(6)}.pfm'),depth)
        idx+=1

    # generate far and camera file
    intrin = get_intrinsic(os.path.join(rawdata_root,'intrinsic','intrinsic_color.txt'))
    poses = []
    extrins = []
    for i in range(len(img_idx)):
        extrin = np.eye(4)
        R = rec_file['rotmats'][i]
        t = rec_file['tvecs'][i]
        extrin[:3, :3] = R
        extrin[:3, 3] = t
        extrins.append(extrin)
        poses.append(np.linalg.pinv(extrin))
    extrins = np.stack(extrins)
    poses = np.stack(poses)

    campos = np.asarray([pose @ np.asarray([0, 0, 0, 1]) for pose in poses])[:, :3]
    pcd_ocam = o3d.geometry.PointCloud()
    pcd_ocam.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(campos))

    # using estimated depth to generate bounding box
    # mesh_pcd = o3d.io.read_point_cloud(os.path.join(preds_root, f'tsdf_voxel2.ply'))
    # mesh_pcd, ind = mesh_pcd.remove_statistical_outlier(nb_neighbors=100,
    #                                               std_ratio=0.1)
    # mesh_pcd.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(mesh_pcd.points), campos]))
    # o3d.visualization.draw_geometries([mesh_pcd])
    # cube,aabb =get_bbox_from_pcd(mesh_pcd,clip=False)

    # # using gt mesh to generate bounding box
    mesh_pcd = o3d.io.read_point_cloud(glob.glob(os.path.join(rawdata_root, f'*_vh_clean_2.ply'))[0])
    mesh_pcd.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(mesh_pcd.points), campos]))
    cube,aabb =get_bbox_from_pcd(mesh_pcd,clip=True)

    o3d.io.write_triangle_mesh(os.path.join(save_root, 'bounding_box.ply'),cube)
    s_scale = 1.05
    aabb.get_extent()
    scale = aabb.get_extent().max() / 2
    meanvec = aabb.get_center()
    x_mean, y_mean, z_mean = meanvec
    test_pcd = mesh_pcd
    test_pcd.points = o3d.utility.Vector3dVector(((np.asarray(test_pcd.points).transpose() - np.asarray(
        [x_mean, y_mean, z_mean]).reshape(3, -1)) / (scale * s_scale)).transpose())

    # visualize optimization space & mesh
    # bbox = get_bbox1()
    # o3d.visualization.draw_geometries([test_pcd, bbox])

    normalization = np.eye(4).astype(np.float32)
    normalization[:3,:3] = normalization[:3,:3] * scale * s_scale
    normalization[:3,3] = meanvec
    print('normalization matrix')
    print(normalization)

    write_cam_npz(normalization, intrin, poses, os.path.join(save_root), 'cameras')
    print('camera file writed')

    # generate far
    ray_len = get_ray_tracing_init(np.asarray(poses),rec_file['K'][0],cube,h=d_h,w=d_w,mesh_f_name_flag=False, depth_flag=False)

    for i in range(ray_len.shape[0]):
        cv2.imwrite(os.path.join(save_root, 'far', f'{i.__str__().zfill(6)}.pfm'), ray_len[i])


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


def preproc_normal(normal_root, preds_root, output, sortfunc=None,index=True):
    '''
    prepare normal and alpha files from surface normal uncertainty output, the filename should not contain '.' ! the output normal is a pfm file contains a h,w,3 matrix, alpha is a pfm file contains a h,w matrix
    :param normal_root: normal uncertainty output root
    :param dv3_recfile: 3dvnet output preds.npz
    :param output: output directory
    :param sort: sort function for a list
    :param index: if output name use idx, or use original name
    :return: None
    '''
    print(f'generate normal priors')
    os.makedirs(os.path.join(output,'normal'),exist_ok=True)
    os.makedirs(os.path.join(output,'alpha'),exist_ok=True)
    nalist = glob.glob(os.path.join(normal_root,'*.npy'))
    rec_file = np.load(os.path.join(preds_root,'preds.npz'))

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


def demo_preproc_normal(normal_root, camera_file, output, sortfunc=None,index=True):
    '''
    prepare normal and alpha files from surface normal uncertainty output, the filename should not contain '.' ! the output normal is a pfm file contains a h,w,3 matrix, alpha is a pfm file contains a h,w matrix
    :param normal_root: normal uncertainty output root
    :param camera_file: camera file cameras.npz
    :param output: output directory
    :param sort: sort function for a list
    :param index: if output name use idx, or use original name
    :return: None
    '''
    print(f'generate normal priors')
    os.makedirs(os.path.join(output,'normal'),exist_ok=True)
    os.makedirs(os.path.join(output,'alpha'),exist_ok=True)
    nalist = glob.glob(os.path.join(normal_root,'*.npy'))
    rec_file = np.load(camera_file)

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