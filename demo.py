

import os
import numpy as np
from functools import partial
from datetime import datetime

import cv2
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from keras_applications.imagenet_utils import preprocess_input

from network import DCNN
from knn_matting import knn_matte
from gen_alpha_from_sharedmatting import shared_matting
models = keras.models
layers = keras.layers
optimizers = keras.optimizers

########### Input Image path
im_path = './test_inputs/imgs/donkey.png'
trimap_path = './test_inputs/trimaps/donkey.png'
output_path = './test_outputs/'
###########


# Model definition
img_size = (512, 512) # width, height
model = DCNN(input_tensor=keras.Input(shape=(img_size[1], img_size[0], 5)), output_shape=(img_size[1], img_size[0]))
model.load_weights('./ckpt/ckpt-.h5')

im = cv2.imread(im_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, dsize=img_size)
trimap = cv2.imread(im_path, cv2.IMREAD_COLOR)
trimap = cv2.resize(trimap, dsize=img_size)

# knn matte
print('Knn matting...')
knn = knn_matte(im, trimap)
knn = knn * 255.
knn = np.clip(knn, 0, 255).astype(np.uint8)
knn = knn[..., None]

# shared matte
print('Shared matting...')
shared = shared_matting(im, trimap)
shared = np.clip(shared, 0, 255).astype(np.uint8)
shared = shared[..., None]

# Predict
print('Predicting...')
c_input = np.concatenate([im, knn, shared], axis=-1)
pred_matte = model.predict([c_input])
pred_matte = pred_matte[0]

### Output 
print('Output')
im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
knn_re = np.dstack([im, knn])
shared_re = np.dstack([im, shared])
pred_re = np.dstack([im, pred_matte])

base_name = os.path.basename(im_path)[:-4]
cv2.imwrite(os.path.join(output_path, base_name + '_knn.png'), knn_re)
cv2.imwrite(os.path.join(output_path, base_name + '_shared.png'), shared_re)
cv2.imwrite(os.path.join(output_path, base_name + '_pred.png'), pred_re)

print('End')






