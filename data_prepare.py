import os
import argparse

import numpy as np
np.random.seed(1)

from config import Config
from utils import mix_by_alpha

parser = argparse.ArgumentParser(description='This python shell prepares the data.')
parser.add_argument('--stage', type=str, help='One of [gen_composite_images, gen_image_patches, train_val_test_split]')

args = parser.parse_args()

def gen_composite_images():
    im_path = os.path.join(Config.ROOT_PATH, Config.train_input_path)
    knn_path = os.path.join(Config.ROOT_PATH, 'knn_{}'.format(Config.train_input_path))
    cf_path = os.path.join(Config.ROOT_PATH, 'closed_form_{}'.format(Config.train_input_path))
    matte_path = os.path.join(Config.ROOT_PATH, Config.gt_train_alpha_path)
    trimap_path = os.path.join(Config.ROOT_PATH, Config.train_trimap_path)
    bg_img_path = Config.bg_image_path

    ims = sorted(os.listdir(im_path))
    knns = sorted(os.listdir(knn_path))
    cfs = sorted(os.listdir(cf_path))
    mattes = sorted(os.listdir(matte_path))
    trimaps = sorted(os.listdir(trimap_path))
    bg_img_names = os.listdir(bg_img_path)

    o_paths = [os.path.join(Config.ROOT_PATH, Config.comp_images_path),
                os.path.join(Config.ROOT_PATH, Config.comp_knn_matte_path),
                os.path.join(Config.ROOT_PATH, Config.comp_closed_form_matte_path),
                os.path.join(Config.ROOT_PATH, Config.comp_matte_path),
                os.path.join(Config.ROOT_PATH, Config.comp_trimap_path)]
    for o_path in o_paths:
        if not os.path.exists(o_path):
            os.makedirs(o_path)
    for i, (im, knn, cf, matte, trimap) in enumerate(zip(ims, knns, cfs, mattes, trimaps)):
        assert im == knn == cf == matte
        if i % 5 == 0:
            print('\r {}/{} th image done...'.format(i, len(ims)), end='')
        np.random.shuffle(bg_img_names)

        _im = cv2.imread(os.path.join(im_path, im), cv2.IMREAD_COLOR)
        _matte = cv2.imread(os.path.join(matte_path, matte), cv2.IMREAD_GRAYSCALE)
        _knn = cv2.imread(os.path.join(knn_path, knn), cv2.IMREAD_GRAYSCALE)
        _cf = cv2.imread(os.path.join(cf_path, cf), cv2.IMREAD_GRAYSCALE)
        _trimap = cv2.imread(os.path.join(trimap_path, trimap), cv2.IMREAD_GRAYSCALE)
        h, w = _im.shape[:2]

        for bg_img in bg_img_names[:Config.comp_ratio]:
            _bg_img = cv2.imread(os.path.join(bg_img_path, bg_img), cv2.IMREAD_COLOR)
            _bg_img = cv2.resize(_bg_img, dsize=(w, h))
            mix_img = mix_by_alpha(_im, _bg_img, return_channel=3, keep_alpha=False)
            comp_name = '{}-{}'.format(im[:-4], bg_img[:-4])

            cv2.imwrite(os.path.join(Config.ROOT_PATH, Config.comp_images_path, comp_name + '.jpg'), mix_img)
            cv2.imwrite(os.path.join(Config.ROOT_PATH, Config.comp_knn_matte_path, comp_name + '.png'), _knn)
            cv2.imwrite(os.path.join(Config.ROOT_PATH, Config.comp_closed_form_matte_path, comp_name + '.png'), _cf)
            cv2.imwrite(os.path.join(Config.ROOT_PATH, Config.comp_matte_path, comp_name + '.png'), _matte)
            cv2.imwrite(os.path.join(Config.ROOT_PATH, Config.comp_trimap_path, comp_name + '.png'), _trimap)

def gen_image_patches():
    im_path = os.path.join(Config.ROOT_PATH, Config.comp_images_path)
    knn_path = os.path.join(Config.ROOT_PATH, Config.comp_knn_matte_path)
    cf_path = os.path.join(Config.ROOT_PATH, Config.comp_closed_form_matte_path)
    matte_path = os.path.join(Config.ROOT_PATH, Config.comp_matte_path)
    trimap_path = os.path.join(Config.ROOT_PATH, Config.comp_trimap_path)

    ims = sorted(os.listdir(im_path))
    knns = sorted(os.listdir(knn_path))
    cfs = sorted(os.listdir(cf_path))
    mattes = sorted(os.listdir(matte_path))
    trimaps = sorted(os.listdir(trimap_path))

    o_paths = [os.path.join(Config.ROOT_PATH, Config.image_patch_path),
                os.path.join(Config.ROOT_PATH, Config.knn_matte_patch_path),
                os.path.join(Config.ROOT_PATH, Config.closed_form_matte_patch_path),
                os.path.join(Config.ROOT_PATH, Config.matte_patch_path)]
    for o_path in o_paths:
        if not os.path.exists(o_path):
            os.makedirs(o_path)

    for i, (im, knn, cf, matte, trimap) in enumerate(zip(ims, knns, cfs, mattes, trimaps)):
        if i % 5 == 0:
            print('\r {}/{} th image done...'.format(i, len(ims)), end='')

        _im = cv2.imread(os.path.join(im_path, im), cv2.IMREAD_COLOR)
        _matte = cv2.imread(os.path.join(matte_path, matte), cv2.IMREAD_GRAYSCALE)
        _knn = cv2.imread(os.path.join(knn_path, knn), cv2.IMREAD_GRAYSCALE)
        _cf = cv2.imread(os.path.join(cf_path, cf), cv2.IMREAD_GRAYSCALE)
        _trimap = cv2.imread(os.path.join(trimap_path, trimap), cv2.IMREAD_GRAYSCALE)
        h, w = _im.shape[:2]
        y_idx, x_idx = np.where(_trimap == Config.unknown_code)
        for j in range(Config.n_patches_one_image):
            randx = np.random.choice(x_idx)
            randy = np.random.choice(y_idx)
            x1 = max(0, int(randx - Config.train_patch_size / 2))
            y1 = max(0, int(randy - Config.train_patch_size / 2))
            x2 = min(w, int(randx + Config.train_patch_size / 2 + 1))
            y2 = min(h, int(randy + Config.train_patch_size / 2 + 1))
            
            im_patch = _im[y1:y2, x1:x2, :]
            matte_patch = _matte[y1:y2, x1:x2]
            knn_patch = _knn[y1:y2, x1:x2]
            cf_patch = _cf[y1:y2, x1:x2]
            patch_name = '{}_{}'.format(im[:-4], j)

            cv2.imwrite(os.path.join(Config.ROOT_PATH, Config.image_patch_path, patch_name + '.jpg'), im_patch)
            cv2.imwrite(os.path.join(Config.ROOT_PATH, Config.knn_matte_patch_path, patch_name + '.png'), knn_patch)
            cv2.imwrite(os.path.join(Config.ROOT_PATH, Config.closed_form_matte_patch_path, patch_name + '.png'), cf_patch)
            cv2.imwrite(os.path.join(Config.ROOT_PATH, Config.matte_patch_path, patch_name + '.png'), matte_patch)


def train_val_test_split():

    im_path = os.path.join(Config.ROOT_PATH, Config.image_patch_path)
    knn_path = os.path.join(Config.ROOT_PATH, Config.knn_matte_patch_path)
    cf_path = os.path.join(Config.ROOT_PATH, Config.closed_form_matte_patch_path)
    matte_path = os.path.join(Config.ROOT_PATH, Config.matte_patch_path)
    
    ims = os.listdir(im_path)
    np.random.shuffle(ims)
    n_train = int(len(ims)*Config.train_val_test_ratio[0]) 
    n_val = int(len(ims)*Config.train_val_test_ratio[1])
    train_ims = ims[:n_train]
    val_ims = ims[n_train:n_train+n_val]
    test_ims = ims[n_train+n_val:]

    def run_stage(c_ims, stage='Train'):
        assert stage in ['Train', 'Validation', 'Test']
        o_im_path = os.path.join(Config.ROOT_PATH, Config.patch_path, stage, Config.image_patch_path)
        o_knn_path = os.path.join(Config.ROOT_PATH, Config.patch_path, stage, Config.knn_matte_patch_path)
        o_cf_path = os.path.join(Config.ROOT_PATH, Config.patch_path, stage, Config.closed_form_matte_patch_path)
        o_matte_path = os.path.join(Config.ROOT_PATH, Config.patch_path, stage, Config.matte_patch_path)

        o_paths = [o_im_path, o_knn_path, o_cf_path, o_matte_path]
        for o_path in o_paths:
            if not os.path.exists(o_path):
                os.makedirs(o_path)
        for c_im in c_ims:
            os.system('cp {} {}'.format(os.path.join(im_path, c_im), o_im_path))
            os.system('cp {} {}.png'.format(os.path.join(knn_path, c_im[:-4]), o_knn_path))
            os.system('cp {} {}.png'.format(os.path.join(cf_path, c_im[:-4]), o_cf_path))
            os.system('cp {} {}.png'.format(os.path.join(matte_path, c_im[:-4]), o_matte_path))

    run_stage(train_ims, stage='Train')
    run_stage(val_ims, stage='Validation')
    run_stage(test_ims, stage='Test')

if __name__ == '__main__':
    if args.stage == 'gen_composite_images':
        gen_composite_images()
    elif args.stage == 'gen_image_patches':
        gen_image_patches()
    elif args.stage == 'train_val_test_split':
        train_val_test_split()
    else:
        raise Exception('This method is not supported yet.')
    