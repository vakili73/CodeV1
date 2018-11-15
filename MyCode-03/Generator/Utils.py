import numpy as np

from skimage import exposure
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate


# %% Util functions

def FlipLR(img):
    return np.fliplr(img)


def FlipUD(img):
    return np.flipud(img)


def Correction(img):
    rnd = np.random.rand()
    if rnd < 0.5:
        return exposure.adjust_log(img)
    elif rnd < 1:
        return exposure.adjust_sigmoid(img)


def PermCH(img):
    noch = img.shape[2] if len(img.shape) > 2 else False
    if noch:
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        colors = np.random.permutation([R, G, B])
        return np.moveaxis(np.array(colors), 0, 2)
    else:
        return img


def RndCrop(img, rate=1.1):
    crop_size = img.shape
    img = imresize(img, rate)
    h, w = img.shape if len(
        img.shape) == 2 else (img.shape[0], img.shape[1])
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    img = img[top:bottom, left:right]
    return img


def RndRotate(img, angle_range=(-15, 15)):
    h, w = img.shape if len(
        img.shape) == 2 else (img.shape[0], img.shape[1])
    angle = np.random.randint(*angle_range)
    img = rotate(img, angle)
    img = imresize(img, (h, w))
    return img


def CutOut(img, s=(0.01, 0.1)):
    img = np.copy(img)
    rnd = np.random.rand()
    if rnd < 0.5:
        mask_value = img.mean()
    else:
        mask_value = 0
    h, w = img.shape if len(
        img.shape) == 2 else (img.shape[0], img.shape[1])
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])
    mask_aspect_ratio = np.random.rand() * 2
    if mask_aspect_ratio < 0.5:
        mask_aspect_ratio += 0.5
    if mask_aspect_ratio > 1.5:
        mask_aspect_ratio -= 0.5
    mask_height = int(np.sqrt(mask_area * mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_aspect_ratio = np.random.rand() * 2
    if mask_aspect_ratio < 0.5:
        mask_aspect_ratio += 0.5
    if mask_aspect_ratio > 1.5:
        mask_aspect_ratio -= 0.5
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1
    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    img[top:bottom, left:right].fill(mask_value)
    return img


def Translate(img, rate=0.1):
    img = np.copy(img)
    HEIGHT, WIDTH = img.shape if len(
        img.shape) == 2 else (img.shape[0], img.shape[1])
    rnd = np.random.rand()
    if rnd < 0.5:
        rnd = np.random.rand()
        if rnd < 0.25:  # Shifting Left
            pix = int(rate*WIDTH)
            img[:, :-pix] = img[:, pix:]
            img[:, -pix:] = 0
        elif rnd < 0.5:  # Shifting Right
            pix = int(rate*WIDTH)
            img[:, pix:] = img[:, :-pix]
            img[:, :pix] = 0
        elif rnd < 0.75:  # Shifting Up
            pix = int(rate*HEIGHT)
            img[:-pix, :] = img[pix:, :]
            img[-pix:, :] = 0
        elif rnd < 1:  # Shifting Down
            pix = int(rate*HEIGHT)
            img[pix:, :] = img[:-pix, :]
            img[:pix, :] = 0
    else :
        rnd = np.random.rand()
        if rnd < 0.25:  # Shifting Up-Left
            pix = int(rate*WIDTH)
            img[:-pix, :-pix] = img[pix:, pix:]
            img[:, -pix:] = 0
            img[-pix:, :] = 0
        elif rnd < 0.5:  # Shifting Up-Right
            pix = int(rate*WIDTH)
            img[:-pix, pix:] = img[pix:, :-pix]
            img[:, :pix] = 0
            img[-pix:, :] = 0
        elif rnd < 0.75:  # Shifting Down-Left
            pix = int(rate*WIDTH)
            img[pix:, :-pix] = img[:-pix, pix:]
            img[:, -pix:] = 0
            img[:pix, :] = 0
        elif rnd < 1:  # Shifting Down-Right
            pix = int(rate*WIDTH)
            img[pix:, pix:] = img[:-pix, :-pix]
            img[:, :pix] = 0
            img[:pix, :] = 0
    return img
