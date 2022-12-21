import cv2
import argparse
from glob import glob
import os

if __name__ == '__main__':
    print('Hello YS')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_in', type=str, default='./')
    parser.add_argument('--root_out', type=str, default='./')
    parser.add_argument('--suffix_in', type=str, default='ppm')
    parser.add_argument('--suffix_out', type=str, default='jpg')
    args = parser.parse_args()

    # root_in = '/home/yswang/Downloads/gitcode/surface_normal_uncertainty/examples'
    # suffix_in = 'ppm'
    # suffix_out = 'jpg'
    images_lis = sorted(glob(os.path.join(args.root_in, f'*.{args.suffix_in}')))
    for id in images_lis:
        picname = id.split('/')[-1]
        picname = picname[:(len(picname)-len(args.suffix_in)-1)]
        pic = cv2.imread(id)
        cv2.imwrite(os.path.join(args.root_out, f'{picname + "." + args.suffix_out}'),pic)

