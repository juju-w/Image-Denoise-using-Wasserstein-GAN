"""
"""
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
GEN_IMG_PATH = 'output/WGAN/fake_noise15'
GEN_CSV = True
