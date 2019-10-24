'''
KNN matting 生成alpha
'''

import os

import numpy as np
import cv2

from closed_form_matting import closed_form_matting_with_trimap
import warnings
warnings.filterwarnings('ignore')

from config import Config

if __name__ == '__main__':

    data_dir = os.path.join(Config.ROOT_PATH, Config.train_input_path)
    trimap_dir = os.path.join(Config.ROOT_PATH, Config.train_trimap_path)
    output_dir = os.path.join(Config.ROOT_PATH, 'closed_form_{}'.format(Config.train_input_path))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Generate matte by closed-form matting.')

    im_names = os.listdir(data_dir)
    for i, im_name in enumerate(im_names):
        print('{}-th image... Total number of images: {}'.format(i, len(im_names)))
        im_path = os.path.join(data_dir, im_name)
        tri_path = os.path.join(trimap_dir, im_name)
        output_path = os.path.join(output_dir, im_name)

        img = cv2.imread(im_path, cv2.IMREAD_COLOR) / 255.0
        trimap = cv2.imread(tri_path, cv2.IMREAD_GRAYSCALE) / 255.0

        alpha = closed_form_matting_with_trimap(img, trimap)
        alpha = alpha * 255.
        alpha = np.clip(alpha, 0, 255).astype(np.uint8)

        cv2.imwrite(output_path, alpha)