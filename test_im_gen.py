
import os
from functools import partial

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from keras_applications.imagenet_utils import preprocess_input


models = keras.models
layers = keras.layers
optimizers = keras.optimizers

from config import Config

# Must set the seed to a fix value
seed = 1

def obtain_data_generators(data_type):
    im_patch_path = os.path.join(Config.ROOT_PATH, Config.patch_path, data_type, Config.image_patch_path)
    matte_patch_path = os.path.join(Config.ROOT_PATH, Config.patch_path, data_type, Config.matte_patch_path)
    knn_patch_path = os.path.join(Config.ROOT_PATH, Config.patch_path, data_type, Config.knn_matte_patch_path)
    cf_patch_path = os.path.join(Config.ROOT_PATH, Config.patch_path, data_type, Config.closed_form_matte_patch_path)

    if data_type == Config.train_patch_path:
        gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=partial(preprocess_input, data_format='channels_last', mode='tf'),
                                             rotation_range=20,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True,
                                             vertical_flip=True)
    else:
        gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=partial(preprocess_input, data_format='channels_last', mode='tf'))
    
    im_gen = gen.flow_from_directory(im_patch_path,
                                          seed=seed,
                                          target_size=(Config.train_patch_size, Config.train_patch_size),
                                          class_mode=None,
                                          batch_size=Config.batch_size)
    
    matte_gen = gen.flow_from_directory(matte_patch_path,
                                          seed=seed,
                                          target_size=(Config.train_patch_size, Config.train_patch_size),
                                          class_mode=None,
                                          color_mode='grayscale',
                                          batch_size=Config.batch_size)
    
    knn_gen = gen.flow_from_directory(knn_patch_path,
                                          seed=seed,
                                          target_size=(Config.train_patch_size, Config.train_patch_size),
                                          class_mode=None,
                                          color_mode='grayscale',
                                          batch_size=Config.batch_size)
    
    cf_gen = gen.flow_from_directory(cf_patch_path,
                                          seed=seed,
                                          target_size=(Config.train_patch_size, Config.train_patch_size),
                                          class_mode=None,
                                          color_mode='grayscale',
                                          batch_size=Config.batch_size)
    return im_gen, matte_gen, knn_gen, cf_gen


train_im_gen, train_matte_gen, train_knn_gen, train_cf_gen = obtain_data_generators(Config.train_patch_path)

# val_im_gen, val_matte_gen, val_knn_gen, val_cf_gen = obtain_data_generators(Config.validation_patch_path)

# test_im_gen, test_matte_gen, test_knn_gen, test_cf_gen = obtain_data_generators(Config.test_patch_path)

im_filenames = train_im_gen.filenames
matte_filenames = train_matte_gen.filenames
knn_filenames = train_knn_gen.filenames
cf_filenames = train_cf_gen.filenames

assert len(im_filenames) == len(matte_filenames) == len(knn_filenames) == len(cf_filenames)

for a, b, c, d in zip(train_im_gen, train_matte_gen, train_knn_gen, train_cf_gen):
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    import sys
    sys.exit(0)