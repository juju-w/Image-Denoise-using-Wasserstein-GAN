# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/2 21:43
@Auth ： kuiju_wang
@File ：test.py
@IDE ：PyCharm

"""
import mesure
from utils import *
from model import generatorNet
from config import *
from image_operation import make_dir

def test():
    make_dir(GEN_IMG_PATH)
    generator = generatorNet()
    generator.build(input_shape=(None, None, None, BATCH_SHAPE[3]))
    generator.load_weights(CHECKPOINT_PATH + 'BEST_ge.parms')

    batch_test_noise = read_img_2_array(TEST_NOISE_PATH)
    batch_test_truth = read_img_2_array(TEST_CLEAN_PATH)
    n_test = len(batch_test_truth)
    fake_img = [generator(tf.convert_to_tensor(val_image)).numpy() for val_image in batch_test_noise]

    name_list = [name.replace('.jpg','.png') for name in os.listdir(TEST_NOISE_PATH)]
    for i in range(n_test):
        cv2.imwrite(os.path.join(GEN_IMG_PATH,name_list[i]), fake_img[i].numpy() * 255.0)

if __name__ == '__main__':
    test()
    if GEN_CSV:
        mesure.mesure()