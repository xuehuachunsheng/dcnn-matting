
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
from functools import partial
from datetime import datetime
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from keras_applications.imagenet_utils import preprocess_input

from network import DCNN

models = keras.models
layers = keras.layers
optimizers = keras.optimizers

from config import Config

# Seed is very important
seed = 1

class DataGenerator(keras.utils.Sequence):
    def __init__(self, mode='train'):
        gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=partial(preprocess_input, data_format='channels_last', mode='tf'))
        if mode == 'train':
            paths = Config.train_patch_path
            gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=partial(preprocess_input, data_format='channels_last', mode='tf'),
                                                                rotation_range=20,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                horizontal_flip=True,
                                                                vertical_flip=True)
        elif mode == 'val':
            paths = Config.validation_patch_path
        elif mode == 'test':
            paths = Config.test_patch_path

        im_patch_path = os.path.join(Config.ROOT_PATH, Config.patch_path, paths, Config.image_patch_path)
        knn_patch_path = os.path.join(Config.ROOT_PATH, Config.patch_path, paths, Config.knn_matte_patch_path)
        cf_patch_path = os.path.join(Config.ROOT_PATH, Config.patch_path, paths, Config.closed_form_matte_patch_path)
        matte_patch_path = os.path.join(Config.ROOT_PATH, Config.patch_path, paths, Config.matte_patch_path)
        
        self.im_gen = gen.flow_from_directory(im_patch_path,
                                                seed=seed,
                                                target_size=(Config.train_patch_size, Config.train_patch_size),
                                                class_mode=None,
                                                batch_size=Config.batch_size)

        self.knn_gen = gen.flow_from_directory(knn_patch_path,
                                                seed=seed,
                                                target_size=(Config.train_patch_size, Config.train_patch_size),
                                                class_mode=None,
                                                color_mode='grayscale',
                                                batch_size=Config.batch_size)

        self.cf_gen = gen.flow_from_directory(cf_patch_path,
                                                seed=seed,
                                                color_mode='grayscale',
                                                target_size=(Config.train_patch_size, Config.train_patch_size),
                                                class_mode=None,
                                                batch_size=Config.batch_size)
        
        self.matte_gen = gen.flow_from_directory(matte_patch_path,
                                                seed=seed,
                                                color_mode='grayscale',
                                                target_size=(Config.train_patch_size, Config.train_patch_size),
                                                class_mode=None,
                                                batch_size=Config.batch_size)
    
    def __len__(self):
        return len(self.im_gen)
    
    def __getitem__(self, index):
        c_im = self.im_gen.__getitem__(index)
        c_knn = self.knn_gen.__getitem__(index)
        c_cf = self.cf_gen.__getitem__(index)
        c_matte = self.matte_gen.__getitem__(index)
        x = np.concatenate([c_im, c_knn, c_cf], axis=-1)
        y = c_matte
        return x, y

train_gen = DataGenerator('train')
val_gen = DataGenerator('val')
test_gen = DataGenerator('test')
model = DCNN()
model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(lr=Config.learning_rate), metrics=[keras.losses.mean_squared_error])
def callbacks():
    def lr_schedule(epoch, learning_rate):
        for x in Config.epoch_decays: # Bug fixed
            if epoch == x:
                return learning_rate * Config.epoch_decay_rate
        return learning_rate
    c_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs/logs-{}'.format(c_time))
    lr_callback = keras.callbacks.LearningRateScheduler(schedule=lr_schedule)
    checkpoint_callback = keras.callbacks.ModelCheckpoint('./ckpt/ckpt-{}.h5'.format(c_time), monitor='val_mean_squared_error', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    callbacks = [tensorboard_callback, lr_callback, checkpoint_callback]
    if Config.early_stop:
        early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', min_delta=0.001, patience=10, mode='min', verbose=1)
        callbacks.append(early_stop_callback)
    return callbacks

callbacks = callbacks()
model.fit_generator(train_gen,
                    steps_per_epoch=len(train_gen),
                    epochs=Config.n_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=val_gen,
                    validation_steps=len(val_gen),
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=True,
                    shuffle=True,
                    initial_epoch=0)