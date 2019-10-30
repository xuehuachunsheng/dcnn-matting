import tensorflow as tf
from tensorflow.python import keras

models = keras.models
layers = keras.layers
optimizers = keras.optimizers

def DCNN(input_tensor=None, output_shape=(27, 27)):
    conv1 = layers.Conv2D(64, (9, 9), padding='valid') # 19x19x64
    conv2 = layers.Conv2D(64, (1, 1), padding='valid') # 19x19x64
    conv3 = layers.Conv2D(64, (1, 1), padding='valid') # 19x19x64
    conv4 = layers.Conv2D(64, (1, 1), padding='valid') # 19x19x64
    conv5 = layers.Conv2D(32, (1, 1), padding='valid') # 19x19x32
    conv6 = layers.Conv2D(1, (5, 5), padding='valid') # 15x15x1
    if input_tensor is None:
        inputs = keras.Input(shape=(27, 27, 5), dtype=tf.float32)
    else:
        inputs = input_tensor
    x = layers.ReLU()(conv1(inputs))
    x = layers.ReLU()(conv2(x))
    x = layers.ReLU()(conv3(x))
    x = layers.ReLU()(conv4(x))
    x = layers.ReLU()(conv5(x))
    x = conv6(x)
    if output_shape is not None:
        x = keras.layers.Lambda(lambda a: tf.image.resize_bicubic(x, size=output_shape))(x)
    model = keras.Model(inputs, x) 
    return model




