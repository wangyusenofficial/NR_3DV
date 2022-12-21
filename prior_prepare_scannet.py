import os
import prior_utils.utils as utils
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannet_root', type=str, default='/home/yswang/data2/scans')
    parser.add_argument('--output', type=str, default='/home/yswang/data2/nr3dv_data')
    parser.add_argument('--scene_list', type=str, default='scenelist')
    parser.add_argument('--n_clean', default=False, action="store_true")
    args = parser.parse_args()

    SCANNET_SCENE_FOLDER = args.scannet_root
    OUTPUT_FOLDER = args.output
    CLEAN = not args.n_clean

    with open(args.scene_list,'r') as f:
        scenes = f.readlines()

    for scene in scenes:
        SCENE = scene.strip()
        # prepare data for 3dvnet
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        os.system(f'python 3dvnet/data_preprocess/preprocess_scannet.py --src {SCANNET_SCENE_FOLDER} --dst {OUTPUT_FOLDER} --scene {SCENE}')
        utils.alter('3dvnet/mv3d/eval/config_template.py','3dvnet/mv3d/eval/config.py','NeedToReplace',OUTPUT_FOLDER)
        os.system(f'python 3dvnet/mv3d/eval-3dvnet.py --scene_path {os.path.join(OUTPUT_FOLDER,SCENE)}')

        # generate neuralroom input from 3dv output
        # generate tsdf for 3dvnet
        print('generate tsdf for 3dvnet')
        utils.tsdf_fusion(preds_root=os.path.join(OUTPUT_FOLDER,'3dvnet','scenes',SCENE), data_root=os.path.join(OUTPUT_FOLDER,SCENE),voxel_size=2,save_root='default')
        print('generate depth prior')
        utils.prepare_data_3dv(preds_root=os.path.join(OUTPUT_FOLDER,'3dvnet','scenes',SCENE), rawdata_root=os.path.join(SCANNET_SCENE_FOLDER,SCENE), save_root=os.path.join(OUTPUT_FOLDER,'nr_'+SCENE))

        # generate normal priors from surface_norm_uncert
        print('generate normal prior')
        os.system(f'cp {os.path.join(OUTPUT_FOLDER,"nr_"+SCENE,"image/*")} surface_normal_uncertainty/examples/')
        os.system(f'cd surface_normal_uncertainty/ ; python test.py --pretrained scannet --architecture BN')
        utils.preproc_normal(normal_root='surface_normal_uncertainty/examples/results', preds_root=os.path.join(OUTPUT_FOLDER,'3dvnet','scenes',SCENE), output=os.path.join(OUTPUT_FOLDER,'nr_'+SCENE))
        os.system(f'rm surface_normal_uncertainty/examples/*.png')
        os.system(f'mv surface_normal_uncertainty/examples/results surface_normal_uncertainty/examples/results_{SCENE}')

        # generate config files for neuralroom
        utils.alter('dataset/config_template.conf',f'dataset/{SCENE}.conf','replace_case_here',os.path.join(OUTPUT_FOLDER,'nr_'+SCENE))

        if CLEAN:
            print('cleaning temporary files')
            os.system(f'rm -rf surface_normal_uncertainty/examples/results_{SCENE}')
            os.system(f'cd {OUTPUT_FOLDER} ; rm -rf 3dvnet/scenes/{SCENE}')
            os.system(f'cd {OUTPUT_FOLDER} ; rm -rf {SCENE}')
            os.system(f'cd {OUTPUT_FOLDER} ; rm -rf 3dvnet')




