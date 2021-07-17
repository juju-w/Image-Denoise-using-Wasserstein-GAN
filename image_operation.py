# -*- coding: utf-8 -*-
"""
@Time ： 2021/6/20 8:38
@Auth ： kuiju_wang
@File ：image_operation.py
@IDE ：PyCharm

"""
import shutil

import cv2
import os
import argparse
from progressbar import ProgressBar
from skimage.util import random_noise
import wget
import tarfile


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, 0o755)


bar = ProgressBar()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''\
    do operation to image (eg. add_noise,resize...)

    
            ''', formatter_class=argparse.RawDescriptionHelpFormatter, usage='''\n
    python image_operation.py -d -i INPUT_DIR -o OUTPUT_DIR -tt_r 0.8       To build dataset,(train data):(test date)=0.8
    python image_operation.py -a -l 25 -i INPUT_DIR -o OUTPUT_DIR           To add levels = 25 gaussian noise to input folder image
    python image_operation.py -r -s 256 -i INPUT_DIR -o OUTPUT_DIR          To resize input folder image to 256*256
            ''')

    parser.add_argument('-a', '--add_noise', action='store_true', help='set add differ level of noise TRUE')
    parser.add_argument('-r', '--resize', action='store_true', help='set resize image to size TRUE')
    parser.add_argument('-d', '--dataset_build', action='store_true',
                        help='set build dataset TRUE,will build train and test folder in targe dir')
    parser.add_argument('-s', '--size', default=256, type=int, help='target image size')
    parser.add_argument('-l', '--noise_levels', default=15, type=int,
                        help='choice the noise level you want to add like 15 or 25 or 50')
    parser.add_argument('-i', '--input_dir', default='./data/0raw/', type=str, help='input image dir')
    parser.add_argument('-o', '--out_dir', default='./data/output/', type=str, help='target dir of operated image')

    args = parser.parse_args()

    origin_dir = args.input_dir
    goal_dir = args.out_dir
    # random.seed(8818)
    if not os.path.exists(origin_dir) and not args.dataset_build:
        print('please check your input dir')
        exit()

    if not args.dataset_build:
        dir_list = os.listdir(origin_dir)

    if not os.path.exists(goal_dir):
        make_dir(goal_dir)

    if args.dataset_build:
        # download bsr500 dataset
        bsd500url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_full.tgz'
        save_path = goal_dir + '/BSR_full.tgz'
        target_path = goal_dir + '/0raw'
        make_dir(target_path)
        if not os.path.exists(save_path):
            print('downloading BSD500 dataset from ' + bsd500url + ' to ' + save_path)
            wget.download(bsd500url, save_path)
        tar_file = tarfile.open(save_path)
        target_file = [t for t in tar_file.getnames() if 'images/' in t and 'BSDS500' in t and '.jpg' in t]
        for i, img in enumerate(target_file):
            tar_file.extract(img, target_path)
            shutil.move(os.path.join(target_path, img), os.path.join(target_path, img.split('/')[-1]))
        shutil.rmtree(target_path + '/BSR/')
        tar_file.close()

        print('done')
        print('start building dataset')
        make_dir(goal_dir + '/1_train/clean')
        make_dir(goal_dir + '/1_train/noise15')
        make_dir(goal_dir + '/1_train/noise25')
        make_dir(goal_dir + '/1_train/noise50')

        make_dir(goal_dir + '/2_val/clean')
        make_dir(goal_dir + '/2_val/noise15')
        make_dir(goal_dir + '/2_val/noise25')
        make_dir(goal_dir + '/2_val/noise50')

        make_dir(goal_dir + '/3_test/clean')
        make_dir(goal_dir + '/3_test/noise15')
        make_dir(goal_dir + '/3_test/noise25')
        make_dir(goal_dir + '/3_test/noise50')

        dir_list = [i.split('/')[-1] for i in target_file]
        # resize image and add noise
        for i in bar(range(len(dir_list))):
            img = cv2.imread(os.path.join(target_path, dir_list[i]))
            x, y = img.shape[0:2]
            if not all([x == args.size, y == args.size]):
                img = cv2.resize(img, dsize=(args.size, args.size))
            if i >= 400:
                tmp_dir = '/3_test'
            elif i < 400 and i >= 350:
                tmp_dir = '/2_val'
            else:
                tmp_dir = '/1_train'
            cv2.imwrite(os.path.join(goal_dir + tmp_dir + '/clean/', dir_list[i]), img)
            for n in [15, 25, 50]:
                noise = random_noise(img / 255.0, mode='gaussian', var=(n / 255.0) ** 2)
                cv2.imwrite(os.path.join(goal_dir + tmp_dir + '/noise' + str(n) + '/', dir_list[i]),
                            (noise * 255.0).astype(int))
        print('noise dataset build finish ,you can find your dataset in < ' + goal_dir + ' >')
        exit()

    if not any([args.resize == 1, args.add_noise == 1]):
        parser.print_help()
        exit()

    for i in bar(range(len(dir_list))):
        img = cv2.imread(os.path.join(origin_dir, dir_list[i]))
        x, y = img.shape[0:2]
        if args.resize:
            if not all([x == args.size, y == args.size]):
                img = cv2.resize(img, dsize=(args.size, args.size))
        if args.add_noise:
            img = random_noise(img, mode='gaussian', var=(args.noise_levels / 255.0) ** 2)
            cv2.imwrite(os.path.join(goal_dir, dir_list[i]), (img * 255.0).astype(int))
        else:
            cv2.imwrite(os.path.join(goal_dir, dir_list[i]), img)
    print('operated image save in folder < ' + goal_dir + ' >')
