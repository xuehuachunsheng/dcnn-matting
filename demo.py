

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
from closed_form_matting import closed_form_matting_with_trimap
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
model.load_weights('./ckpt/ckpt-20191024-152441.h5')

im = cv2.imread(im_path, cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, dsize=img_size)
trimap = cv2.imread(trimap_path, cv2.IMREAD_COLOR)
trimap = cv2.resize(trimap, dsize=img_size)

# knn matte
print('Closed-form matting...(But this method is not stable, so i do not use it to test)')
# img_norm = im / 255.0
# trimap_norm = cv2.cvtColor(trimap, cv2.COLOR_BGR2GRAY) / 255.0
# cf = closed_form_matting_with_trimap(img_norm, trimap_norm)
# cf = cf * 255.
# cf = np.clip(cf, 0, 255).astype(np.uint8)
# cf = cf[..., None]

# shared matte
print('Shared matting...')
shared = shared_matting(im, trimap)
shared = np.clip(shared, 0, 255).astype(np.uint8)
shared = shared[..., None]

# Predict
print('Predicting...')
cf = np.array(shared, copy=False)
c_input = np.concatenate([im, shared, cf], axis=-1) # Replace the knn matting.
pred_matte = model.predict(c_input[None, ...])
pred_matte = pred_matte[0]

### Output 
print('Output')
im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
cf_re = np.dstack([im, cf])
shared_re = np.dstack([im, shared])
pred_re = np.dstack([im, pred_matte])

base_name = os.path.basename(im_path)[:-4]
cv2.imwrite(os.path.join(output_path, base_name + '_cf.png'), cf_re)
cv2.imwrite(os.path.join(output_path, base_name + '_shared.png'), shared_re)
cv2.imwrite(os.path.join(output_path, base_name + '_pred.png'), pred_re)

print('End')






