
import os, sys
import cv2

from config import Config
def test():
    img_patch_path = os.path.join(Config.ROOT_PATH, Config.image_patch_path)
    matte_patch_path = os.path.join(Config.ROOT_PATH, Config.matte_patch_path)
    knn_patch_path = os.path.join(Config.ROOT_PATH, Config.knn_matte_patch_path)
    cf_patch_path = os.path.join(Config.ROOT_PATH, Config.closed_form_matte_patch_path)

    ims = os.listdir(img_patch_path)
    ims = sorted(ims)
    mattes = os.listdir(matte_patch_path)
    mattes = sorted(mattes)
    knns = os.listdir(knn_patch_path)
    knns = sorted(knns)
    cfs = os.listdir(cf_patch_path)
    cfs = sorted(cfs)

    for i in range(3):
        im = cv2.imread(os.path.join(img_patch_path, ims[i]))
        matte = cv2.imread(os.path.join(matte_patch_path, mattes[i]), cv2.IMREAD_GRAYSCALE)
        knn = cv2.imread(os.path.join(knn_patch_path, knns[i]), cv2.IMREAD_GRAYSCALE)
        cf = cv2.imread(os.path.join(cf_patch_path, cfs[i]), cv2.IMREAD_GRAYSCALE)

        cv2.imwrite('../test_outputs/im_{}'.format(ims[i]), im)
        cv2.imwrite('../test_outputs/matte_{}.png'.format(ims[i][:-4]), matte)
        cv2.imwrite('../test_outputs/knn_{}.png'.format(ims[i][:-4]), knn)
        cv2.imwrite('../test_outputs/cf_{}.png'.format(ims[i][:-4]), cf)



if __name__ == '__main__':
    test()


