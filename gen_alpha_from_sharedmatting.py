'''
KNN matting 生成alpha,
'''

import os, sys
import numpy as np
import cv2
from platform import system
from ctypes import CDLL, c_int, POINTER, c_uint8
from config import Config

cdll_names = {
        'Darwin' : './sharedmatting/sharedmatting.dylib',
        'Linux'  : './sharedmatting/sharedmatting.so',
        'Windows': './msvcrt.dll'
}

_dll = CDLL(cdll_names[system()])
_sharedMatting = _dll.sharedMatting


def shared_matting(im_nd, trimap_nd):
    '''
    :param im_nd: uint8 ndarray
    :param trimap_nd: uint8 ndarray
    :return:
    '''
    # convert to pointer type
    im_nd = np.asarray(im_nd, dtype=np.uint8)
    trimap_nd = np.asarray(trimap_nd, dtype=np.uint8)
    im_nd_pp = im_nd.ctypes.data_as(POINTER(c_uint8))
    trimap_nd_pp = trimap_nd.ctypes.data_as(POINTER(c_uint8))
    matting = np.zeros(im_nd.shape[:2]).astype(np.uint8)
    matting_pp = matting.ctypes.data_as(POINTER(c_uint8))
    height = c_int(im_nd.shape[0]) # height
    width = c_int(im_nd.shape[1]) # width
    _sharedMatting(im_nd_pp, trimap_nd_pp, matting_pp, height, width)
    return matting


if __name__ == '__main__':

    data_dir = os.path.join(Config.ROOT_PATH, Config.train_input_path)

    trimap_dir = os.path.join(Config.ROOT_PATH, Config.train_trimap_path)

    output_dir = os.path.join(Config.ROOT_PATH, 'knn_{}'.format(Config.train_input_path))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Generate matte by shared matting (replace the knn matting).')
    im_names = os.listdir(data_dir)

    for i, im_name in enumerate(im_names):
        output_path = os.path.join(output_dir, im_name)
        if os.path.exists(output_path):
            continue
        print('Current Image: {}... Total number of images: {}'.format(im_name, len(im_names)))
        im_path = os.path.join(data_dir, im_name)
        tri_path = os.path.join(trimap_dir, im_name)
        img = cv2.imread(im_path, cv2.IMREAD_COLOR)
        trimap = cv2.imread(tri_path, cv2.IMREAD_COLOR)
        assert img.dtype == trimap.dtype == np.uint8
        alpha = shared_matting(img, trimap)
        alpha = np.clip(alpha, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, alpha)
