# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/21 16:32
@Auth ： kuiju_wang
@File ：main.py
@IDE ：PyCharm

"""
import numpy as np

from model import generatorNet, discriminatorNet
from utils import *
from progressbar import *
from skimage.metrics import *

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

epoch_bar = ProgressBar


def train():
    truth = read_img_2_array(TRAIN_CLEAN_PATH)
    noise = read_img_2_array(TRAIN_NOISE_PATH)
    truth, noise = get_patch(truth, noise)

    batch_val_truth = read_img_2_array(VAL_CLEAN_PATH)
    batch_val_noise = read_img_2_array(VAL_NOISE_PATH)

    generator = generatorNet()
    generator.build(input_shape=(None, None, None, BATCH_SHAPE[3]))

    discriminator = discriminatorNet()
    discriminator.build(input_shape=(None, PATCH_SHAPE[1], PATCH_SHAPE[2], PATCH_SHAPE[3]))

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=G_LEARNING_RATE, name="g_optimizer")
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LEARNING_RATE, name="d_optimizer")

    max_ssmi = 0
    max_psnr = 0
    min_mse = 0

    for epoch in range(N_EPOCHS):

        for times in epoch_bar(range(truth.shape[0] // BATCH_SIZE)):

            batch_truth = tf.convert_to_tensor(truth[BATCH_SIZE * times:BATCH_SIZE * (times + 1)])
            batch_noise = tf.convert_to_tensor(noise[BATCH_SIZE * times:BATCH_SIZE * (times + 1)])

            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_noise, batch_truth)
                d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                g_loss = g_loss_fn(generator, discriminator, batch_noise, batch_truth)
                g_grads = tape.gradient(g_loss, generator.trainable_variables)
                g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

            # evluate
            fake_img = [generator(tf.convert_to_tensor(val_image)).numpy() for val_image in batch_val_noise]
            imgraw = batch_val_truth
            n_val = len(imgraw)
            psnr = np.mean(np.array([peak_signal_noise_ratio(imgraw[n], fake_img[n]) for n in range(n_val)]))
            ssim = np.mean(np.array([structural_similarity(imgraw[n], fake_img[n], multichannel=True, data_range=1) for n in range(n_val)]))
            mse = np.mean(np.array([mean_squared_error(imgraw[n], fake_img[n]) for n in range(n_val)]))
            print('EPOCH:' + str(epoch) + ",  d_loss:" + str(round(d_loss.numpy(), 6)) + ",  g_loss:" + str(
                round(g_loss.numpy(), 6)) + ', ssim:' + str(ssim) + ', psnr:' + str(psnr) + ', MSE:' + str(mse))
            if epoch > 1 and ssim > max_ssmi and psnr > max_psnr and mse < min_mse:
                max_ssmi,max_psnr,min_mse = ssim,psnr,mse
                generator.save_weights(CHECKPOINT_PATH + 'BEST_ge.parms')
                if SAVE_DIS_WEIGHT:
                    discriminator.save_weights(CHECKPOINT_PATH + 'BEST_di.parms')
            elif epoch in [N_EPOCHS // t for t in range(5)]:
                generator.save_weights(CHECKPOINT_PATH + 'EPOCH_' + str(epoch) + '_ge.parms')
                if SAVE_DIS_WEIGHT:
                    discriminator.save_weights(CHECKPOINT_PATH + 'EPOCH_' + str(epoch) + '_di.parms')

    generator.save_weights(CHECKPOINT_PATH + 'FINAL_ge.parms')
    discriminator.save_weights(CHECKPOINT_PATH + 'FINAL_di.parms')


if __name__ == '__main__':
    train()
