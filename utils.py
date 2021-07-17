# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/21 16:05
@Auth ： kuiju_wang
@File ：utils.py.py
@IDE ：PyCharm

"""
import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.losses import MSE
from skimage.filters import gaussian
import random
from config import *


def batch_ssim_psnr_sum(batch_turth, batch_fake):
    ssim = tf.reduce_mean(tf.image.ssim(batch_turth, batch_fake, 1.0))
    ssim = tf.multiply(ssim, SSIM_FACTOR)
    psnr = tf.reduce_mean(tf.image.psnr(batch_turth, batch_fake, 1.0))
    psnr = tf.multiply(psnr, PSNR_FACTOR)
    return tf.add(ssim, psnr)


def batch_ssim_psnr_show(batch_turth, batch_fake):
    ssim = tf.reduce_mean(tf.image.ssim(batch_turth, batch_fake, 1.0))
    psnr = tf.reduce_mean(tf.image.psnr(batch_turth, batch_fake, 1.0))
    return ssim, psnr


def img_cut_resize():
    root_dir = './data/bsds500/origal'
    goal_dir = './data/bsds500/256_cut'

    dir_list = os.listdir(root_dir)

    for name in dir_list:
        flag = 0
        img = cv2.imread(os.path.join(root_dir, name))
        imshape = img.shape
        if [imshape[0], imshape[1]] == [321, 481]:
            flag = 1
        if flag:
            img = cv2.flip(cv2.transpose(img), 0)
        img = img[80:401, :, :]
        if flag:
            img = cv2.flip(cv2.transpose(img), 1)
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(os.path.join(goal_dir, name), img)


def zero_one_scale(array_high_dim, tenor=False):
    num = array_high_dim.shape[0]
    dim = array_high_dim.shape[-1]
    for n in range(num):
        for d in range(dim):
            array = array_high_dim[n][:, :, d]
            max_num = np.max(array)
            min_num = np.min(array)
            array_high_dim[n][:, :, d] = (array - min_num) / (max_num - min_num)
    if tenor:
        return tf.convert_to_tensor(array_high_dim)
    else:
        return array_high_dim


def tensor_normalization(tenor):
    num = BATCH_SIZE
    dim = tenor.shape[-1]
    for n in range(num):
        for d in range(dim):
            array = tenor[n][:, :, d]
            max_num = tf.reduce_max(array)
            min_num = tf.reduce_min(array)
            if d == 0:
                x = tf.divide(tf.subtract(array, min_num), (max_num - min_num))
                x = tf.reshape(x, (x.shape[0], x.shape[1], 1))
                xshape = x.shape
            else:
                x = tf.concat([x, tf.reshape(tf.divide(tf.subtract(array, min_num), (max_num - min_num)), xshape)], -1)
        if n == 0:
            y = x
            y = tf.reshape(y, (1, y.shape[0], y.shape[1], y.shape[2]))
            yshape = y.shape
        else:
            y = tf.concat([y, tf.reshape(x, yshape)], 0)
    return y


def gausseimg(tenor):
    image = tensor_normalization(tenor)
    for n in range(BATCH_SIZE):
        img = image[n].eval()
        gauss_out = gaussian(img, sigma=5, multichannel=True)
        img_out = img - gauss_out + 127.0
        img_out = img_out / 255.0
        # 饱和处理
        mask_1 = img_out < 0
        mask_2 = img_out > 1

        img_out = img_out * (1 - mask_1)
        img_out = img_out * (1 - mask_2) + mask_2

        image[n] = tf.convert_to_tensor(Soft_light(img_out, img))
    return image


def Soft_light(img_1, img_2):
    mask = img_1 < 0.5
    T1 = (2 * img_1 - 1) * (img_2 - img_2 * img_2) + img_2
    T2 = (2 * img_1 - 1) * (np.sqrt(img_2) - img_2) + img_2
    img = T1 * mask + T2 * (1 - mask)
    return img


def add_gauss_noise2img(mean, var, image=None):
    if image is None:
        root_dir = 'S:/study/graduatio_design/code_work/data/data_use/256_cut_origal'
        goal_dir = 'S:/study/graduatio_design/code_work/data/data_use/var_15'

        dir_list = os.listdir(root_dir)

        for name in dir_list:
            img = cv2.imread(os.path.join(root_dir, name))
            img = np.array(img / 255, dtype=float)
            np.random.seed(666)
            noise = np.random.normal(mean, (var / 255.0) ** 2, img.shape)
            img = img + noise
            if img.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            img = np.clip(img, low_clip, 1.0)
            img = np.uint8(img * 255)

            cv2.imwrite(os.path.join(goal_dir, name), img)
    else:
        img = image.astype(np.float32)
        noise = np.random.normal(mean, var ** 0.5, img.shape)
        img = img + noise
        return zero_one_scale(img)


def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs


def dataset_truth_build():
    print("reading raw image dataset")
    origin_dir = TRAIN_CLEAN_PATH
    dir_list = os.listdir(origin_dir)
    batch = (np.array([np.array(cv2.imread(os.path.join(origin_dir, name))) for name in dir_list]).astype(
        np.float32)) / 255.0
    print('done')
    return batch


def dataset_noise_build():
    print("reading noise dataset")
    origin_dir = TRAIN_NOISE_PATH
    dir_list = os.listdir(origin_dir)
    batch = (np.array([np.array(cv2.imread(os.path.join(origin_dir, name))) for name in dir_list]).astype(
        np.float32)) / 255.0
    print('done')
    return batch


def get_patch(raw, noise, patch_num=PATCH_NUM, patch_size=PATCH_SHAPE[1]):
    out_raw = []
    out_noise = []
    max_x_y = raw.shape[1] - patch_size
    print("generating patches")
    for n in range(raw.shape[0]):
        for pn in range(patch_num):
            rx = random.randint(0, max_x_y)
            ry = random.randint(0, max_x_y)
            rf = random.choice([-1, 0, 1, None])
            if rf is not None:
                out_raw.append(cv2.flip(raw[n], rf)[rx:rx + patch_size, ry:ry + patch_size, :])
                out_noise.append(cv2.flip(noise[n], rf)[rx:rx + patch_size, ry:ry + patch_size, :])
            else:
                out_raw.append(raw[n][rx:rx + patch_size, ry:ry + patch_size, :])
                out_noise.append(noise[n][rx:rx + patch_size, ry:ry + patch_size, :])
    SEED = np.random.randint(0, 10000)
    np.random.seed(SEED)
    np.random.shuffle(out_raw)
    np.random.seed(SEED)
    np.random.shuffle(out_noise)
    print('done')
    return np.array(out_raw), np.array(out_noise)


def loss_ones(logits):
    labels = tf.ones_like(logits)
    loss = tf.keras.losses.binary_crossentropy(y_true=labels, y_pred=logits, from_logits=True)
    return tf.reduce_mean(loss)


def loss_zeros(logits):
    labels = tf.zeros_like(logits)
    loss = tf.keras.losses.binary_crossentropy(y_true=labels, y_pred=logits, from_logits=True)
    return tf.reduce_mean(loss)


def d_loss_fn(generator, discriminator, batch_noise, batch_truth):
    batch_fake = generator(batch_noise)
    d_fake_logits = discriminator(batch_fake)
    d_truth_logits = discriminator(batch_truth)
    d_loss_fake = loss_zeros(d_fake_logits)
    d_loss_truth = loss_ones(d_truth_logits)
    return tf.multiply(tf.add(d_loss_fake, d_loss_truth), D_LOSS_FACTOR)


def g_loss_fn(generator, discriminator, batch_noise, batch_truth):
    batch_fake = generator(batch_noise)
    g_fake_logits = discriminator(batch_fake)

    SPLOSS = tf.multiply(batch_ssim_psnr_sum(batch_truth, batch_fake), SP_LOSS_FACTOR)
    WLoss = tf.multiply(loss_ones(g_fake_logits), ADVERSARIAL_LOSS_FACTOR)
    PLoss = tf.multiply(get_pixel_loss(batch_truth, batch_fake), PIXEL_LOSS_FACTOR)
    SLoss = tf.multiply(get_smooth_loss(batch_fake), SMOOTH_LOSS_FACTOR)

    loss = tf.add(tf.add(WLoss, PLoss), tf.add(SLoss, SPLOSS))

    return loss


def RGB_TO_BGR(img):
    img_channel_swap = img[..., ::-1]
    # img_channel_swap_1 = tf.reverse(img, axis=[-1])
    return img_channel_swap


def get_pixel_loss(target, prediction):
    return tf.reduce_sum(MSE(target, prediction))


def get_smooth_loss(image):
    batch_count = tf.shape(image)[0]
    image_height = tf.shape(image)[1]
    image_width = tf.shape(image)[2]

    horizontal_normal = tf.slice(image, [0, 0, 0, 0], [batch_count, image_height, image_width - 1, 3])
    horizontal_one_right = tf.slice(image, [0, 0, 1, 0], [batch_count, image_height, image_width - 1, 3])
    vertical_normal = tf.slice(image, [0, 0, 0, 0], [batch_count, image_height - 1, image_width, 3])
    vertical_one_right = tf.slice(image, [0, 1, 0, 0], [batch_count, image_height - 1, image_width, 3])
    smooth_loss = tf.nn.l2_loss(horizontal_normal - horizontal_one_right) + tf.nn.l2_loss(
        vertical_normal - vertical_one_right)
    return smooth_loss

def read_img_2_array(path):
    dir_list = os.listdir(path)
    batch = (np.array([np.array(cv2.imread(os.path.join(path, name))) for name in dir_list]).astype(
        np.float32)) / 255.0
    return batch

def val_truth():
    root_dir = VAL_CLEAN_PATH
    dir_list = os.listdir(root_dir)
    batch = (np.array([np.array(cv2.imread(os.path.join(root_dir, name))) for name in dir_list]).astype(
        np.float32)) / 255.0
    return batch


def val_noise():
    root_dir = VAL_NOISE_PATH
    dir_list = os.listdir(root_dir)
    batch = (np.array([np.array(cv2.imread(os.path.join(root_dir, name))) for name in dir_list]).astype(
        np.float32)) / 255.0
    return tf.convert_to_tensor(batch)


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x
