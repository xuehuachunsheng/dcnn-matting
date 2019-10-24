
import os, sys
import cv2

from config import Config
def test():
    comp_im_path = os.path.join(Config.ROOT_PATH, Config.comp_images_path)
    comp_knn_path = os.path.join(Config.ROOT_PATH, Config.comp_knn_matte_path)

    print(comp_im_path)
    print(comp_knn_path)

    ims = os.listdir(comp_im_path)
    ims = sorted(ims)
    knns = os.listdir(comp_knn_path)
    knns = sorted(knns)

    for i in range(3):
        im = cv2.imread(os.path.join(comp_im_path, ims[i]))
        knn = cv2.imread(os.path.join(comp_knn_path, knns[i]), cv2.IMREAD_GRAYSCALE)

        cv2.imwrite('../test_outputs/im_{}'.format(ims[i]), im)
        cv2.imwrite('../test_outputs/knn_{}.png'.format(ims[i][:-4]), knn)



if __name__ == '__main__':
    test()


