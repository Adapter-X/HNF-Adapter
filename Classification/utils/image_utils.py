import numpy as np
import pickle
from PIL import Image

from data.augment import augment_train, augment_val
import cv2

# 判断文件结尾是否为.npy
def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])


def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])


def load_img(clean_filepath, noisy_filepath, is_train, target_size):
    clean_img = Image.open(clean_filepath).convert('RGB')
    noisy_img = Image.open(noisy_filepath).convert('RGB')
    if is_train:
        clean_img, noisy_img = augment_train(clean_img, noisy_img, target_size)
    else:
        clean_img, noisy_img = augment_val(clean_img, noisy_img, target_size)

    return clean_img, noisy_img


def reorder_image(img, input_order='HWC'):
        """Reorder images to 'HWC' order.

        If the input_order is (h, w), return (h, w, 1);
        If the input_order is (c, h, w), return (h, w, c);
        If the input_order is (h, w, c), return as it is.

        Args:
            img (ndarray): Input image.
            input_order (str): Whether the input order is 'HWC' or 'CHW'.
                If the input image shape is (h, w), input_order will not have
                effects. Default: 'HWC'.

        Returns:
            ndarray: reordered image.
        """

        if input_order not in ['HWC', 'CHW']:
            raise ValueError(
                f'Wrong input_order {input_order}. Supported input_orders are '
                "'HWC' and 'CHW'")
        if len(img.shape) == 2:
            img = img[..., None]
        if input_order == 'CHW':
            img = img.transpose(1, 2, 0)
        return img

def save_img(filenames, img, pred, gt):

    img = img.permute(1, 2, 0).cpu().detach().numpy()
    pred = pred.permute(1, 2, 0).cpu().detach().numpy()
    gt = gt.permute(1, 2, 0).cpu().detach().numpy()

    img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    pred = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    gt = cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.imwrite(filenames + '_img.png', img)
    cv2.imwrite(filenames + '_pred.png', pred)
    cv2.imwrite(filenames + '_gt.png', gt)