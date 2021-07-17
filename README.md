# Image Denoise using Wasserstein-GAN

[中文](md/zh.md)
This is my undergraduate design  for my Degree of Bachelor ___Research on Image Denoising Method Based on Generative Adversarial Network___

This design is based on this [article](https://uofi.box.com/shared/static/s16nc93x8j6ctd0ercx9juf5mqmqx4bp.pdf) | [github](https://github.com/manumathewthomas/ImageDenoisingGAN)

And was inspired by this [github](https://github.com/iteapoy/GANDenoising)

tensorflow2.0 GAN code was inspired [github](https://github.com/huzixuan1/TF_2.0/tree/master/GAN)

other denoising way [github](https://github.com/wenbihan/reproducible-image-denoising-state-of-the-art)

## __Generate Network:__

![generate network](md/generate network.jpg)

The generation network is divided into three parts:

- conv layer : extracting image/noise features
- residual block: using ___short cut___ to accelerate model training, to solve vanishing gradient problem
- deconv layer: Up-sampling get the learning noise

<img src="md/image flow.jpg" alt="image flow" style="zoom:50%;" />

<center> image flow in generate network </center>

## __Discriminate network:__

![discriminate network](md/discriminate network.jpg)

According to the main idea of WGAN, change the last ___sigmoid layer___ to ___dense layer___  convert into solving __Regression Issues__

## __Improvements:__

- Add ___Self-attention-like___ multiply pathway in generate network
- Improved ___LOSS function___ of the generative network
- ___Increased stability___ of model training with Wasserstein-GAN

## Results:

![baboon](md/baboon.jpg)

![snow house](md/snow house.jpg)

![image vs noise 25 ](md/image vs noise 25 .jpg)

<center>noise level 25 differ denoise way vs each others </center>

<img src="md/noise level 15 .png" alt="noise level 15 " style="zoom:79%;" />

<center>noise level 15</center>

<img src="md/noise level 25 .png" alt="noise level 25 " style="zoom:75%;" />

<center>noise level 25</center>

<img src="md/noise level 50 .png" alt="noise level 50 " style="zoom:74%;" />

<center>noise level 50</center>

## __Quick Start:__

#### Requirements:

- python == 3.8.10
- tensorflow == 2.3.0
- opencv  == 4.0.1
- scikit-image == 0.18.1
- numpy == 1.20.2
- pandas == 1.2.5

(optional but recommend)

- cuda == 10.1
- cudnn == 7.6.5
- wget == 3.2 

```shell
# use conda to solve python environment
conda create -n wgan python=3.8
conda activate wgan
conda install cudatoolkit=10.1 cudnn=7.6.5 tensorflow==2.3.0 numpy opencv scikit-image
```

#### Download code from github

```shell
git https://github.com/juju-w/Image-Denoise-using-Wasserstein-GAN.git
cd Image-Denoise-using-Wasserstein-GAN
```

#### Build your own dataset

```shell
# to get help
python image_operation.py -h

# first generate your own dataset with image_operation.py automatically
python image_operation.py \
		--dataset_build \
		--input_dir <INPUT_FOLDER> \
		--out_dir <OUT_FOLDER> 

# or build by yourself manually
python image_operation.py \
		--add_noise --noise_levels 15 \
		--resize --size 256 \
		--input_dir INPUT_FOLDER \
		--out_dir <OUT_FOLDER>
```

####  Edit config file  `config.py`

```python
"""
TRAIN CONFIG
"""
D_LEARNING_RATE = 0.0001	# Discriminater learning rate
G_LEARNING_RATE = 0.0001	# Generater learning rate
BATCH_SIZE = 64		# batch size
PATCH_NUM = 128		# patch per image
PATCH_SHAPE = [BATCH_SIZE, 64, 64, 3]		# pathc size
BATCH_SHAPE = [BATCH_SIZE, 256, 256, 3]		# bathc size
N_EPOCHS = 20		# epoch num
SAVE_DIS_WEIGHT = False     # IF SAVE DISCIMINATER WEIGHT
# LOSS weight factor
ADVERSARIAL_LOSS_FACTOR = 1.0
PIXEL_LOSS_FACTOR = 0.001
STYLE_LOSS_FACTOR = 0
SP_LOSS_FACTOR = 0.5
SMOOTH_LOSS_FACTOR = 0
SSIM_FACTOR = - 20.0
PSNR_FACTOR = - 2.0
D_LOSS_FACTOR = 1.0
# PATH
TRAIN_CLEAN_PATH = 'data/output/1_train/clean/'
TRAIN_NOISE_PATH = 'data/output/1_train/noise15/'
VAL_CLEAN_PATH = 'data/output/2_val/clean/'
VAL_NOISE_PATH = 'data/output/2_val/noise15/'
TEST_CLEAN_PATH = 'data/output/3_test/clean/'
TEST_NOISE_PATH = 'data/output/3_test/noise15/'
CHECKPOINT_PATH = 'checkpoint/noise15/'
"""
TEST CONFIG
"""
GEN_IMG_PATH = 'output/WGAN/fake_noise15'	#faking img save path
GEN_CSV = True		# genrate index csv file after test 

```

#### RUN

```shell
# train 
python train.py

# test 
python test.py
```

#### MESURE

find csv file in `GEN_IMG_PATH.csv` default in `output/WGAN/fake_noise15.csv`

## Citing 

If you use  this WGAN image denoise way in your research, please consider use the following BibTeX entry.

```
@misc{juju-w2021WGAN,
  author = {juju-w},
  title = {Image-Denoise-using-Wasserstein-GAN},
  year = {2021},
  howpublished = {\url{https://github.com/juju-w/Image-Denoise-using-Wasserstein-GAN}}
}
```

