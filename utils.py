
import cv2
import os

def mix_by_alpha(img_1, img_2, return_channel, keep_alpha, inplace=False):
    """
    img_1覆盖在img2上，对于alpha值的混合:
        * keep_alpha=False，即进行alpha值覆盖，即使用前景图片img_1的alpha（ps:若img_1的alpha为0，则使用img_2的alpha值)
        * keep_alpha=True，即使用背景图片img_2的alpha值

    :param img_1: 混合后在上层的图像
    :param img_2: 混合后在下层的图像
    :param return_chanel: 返回图像通道数 int (3 or 4)
    :param keep_alpha: 是否进行alpha通道混合， true-保持img_2的alpha通道不变
    :return: 混合后的图片
    """
    img = img_2 if inplace else img_2.copy()
    alpha = (img_1[:, :, 3] / 255)[:, :, None]
    img[:, :, :3] = img_1[:, :, :3] * alpha + img_2[:, :, :3] * (1 - alpha)

    if return_channel == 4:
        if not keep_alpha:
            idx = img_1[:, :, 3] != 0
            img[:, :, 3][idx] = img_1[:, :, 3][idx]
    else:
        img = img[:, :, :3]

    return img

