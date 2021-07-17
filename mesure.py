# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/15 15:04
@Auth ： kuiju_wang
@File ：mesure.py.py
@IDE ：PyCharm

"""
from skimage.metrics import *
import cv2
import numpy as np
import os
import pandas as pd
from config import *
from progressbar import *

raw_dir = TEST_CLEAN_PATH
test_dir = GEN_IMG_PATH

def measure_psrn_ssim(imgraw, imgmesure):
    psnr = peak_signal_noise_ratio(imgraw, imgmesure)
    ssim = structural_similarity(imgraw, imgmesure, multichannel=True, data_range=255)
    mse = mean_squared_error(imgraw, imgmesure)
    return psnr, ssim, mse


def mesure():
    out_csv = pd.DataFrame(columns=['name', 'PSNR', 'SSIM', 'MSE'])
    rawlist = sorted(os.listdir(raw_dir))
    test_list = sorted(os.listdir(test_dir))
    imgraw = np.array([np.array(cv2.imread(os.path.join(raw_dir, name))) for name in rawlist])
    imgmesure = np.array([np.array(cv2.imread(os.path.join(test_dir, name))) for name in test_list])
    bar = ProgressBar()
    for i in bar(range(imgmesure.shape[0])):
        psnr, ssim, mse = measure_psrn_ssim(imgraw[i], imgmesure[i])
        out_csv.loc[i] = {'name': test_list[i], 'PSNR': psnr, 'SSIM': ssim, 'MSE': mse}
    out_csv.to_csv(test_dir + '.csv')


