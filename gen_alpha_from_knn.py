'''
KNN matting 生成alpha,
'''

import os, sys

import imageio
import scipy.misc
import numpy as np
from knn_matting import knn_matte


from config import Config

if __name__ == '__main__':

    data_dir = os.path.join(Config.ROOT_PATH, Config.train_input_path)

    trimap_dir = os.path.join(Config.ROOT_PATH, Config.train_trimap_path)

    output_dir = os.path.join(Config.ROOT_PATH, 'knn_{}'.format(Config.train_input_path))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Generate matte by knn matting.')
    im_names = os.listdir(data_dir)

    for i, im_name in enumerate(im_names):
        output_path = os.path.join(output_dir, im_name)
        if os.path.exists(output_path):
            continue
        print('Current Image: {}... Total number of images: {}'.format(im_name, len(im_names)))
        im_path = os.path.join(data_dir, im_name)
        tri_path = os.path.join(trimap_dir, im_name)

        img = imageio.imread(im_path)[:, :, :3]
        img = np.asarray(img)
        trimap = imageio.imread(tri_path)
        trimap = np.asarray(trimap)

        if len(trimap.shape) == 2:
            trimap = np.dstack([trimap, trimap, trimap])
        alpha = knn_matte(img, trimap)
        alpha = alpha * 255.
        alpha = np.clip(alpha, 0, 255).astype(np.uint8)
        imageio.imsave(output_path, alpha)
