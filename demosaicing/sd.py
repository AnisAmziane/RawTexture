# Spectral difference MSFA demosaicing

import numpy as np
from scipy.ndimage import convolve

def to8Bit(I):
    mn = I.min()
    mx = I.max()
    mx-=mn
    I = ((I-mn)/mx)*255
    return I.astype(np.uint8)


def toFloat32(I):
    I = I/np.max(I)
    return I.astype(np.float32)

def get_H(pattern):
    NB_rows = pattern.shape[0]
    NB_cols = pattern.shape[1]
    vr = np.array([np.arange(1, NB_rows + 1).tolist() + np.arange(NB_rows - 1, 0, -1).tolist()]).T
    vc = np.array([np.arange(1, NB_cols + 1).tolist() + np.arange(NB_cols - 1, 0, -1).tolist()])
    return (vr * vc) / (vr.max() * vc.max())

def create_pattern_image(pattern, width, height):
    w, h = pattern.shape
    pattern_image = np.zeros((width, height), dtype=np.uint8)
    for l in range(0, width, w):
        for c in range(0, height, h):
            tmp = pattern_image[l:(l + w), c:(c + h)]
            if tmp.shape[:2] == (width, height):
                pattern_image[l:(l + w), c:(c + h)] = pattern
            else:
                xx, yy = pattern_image[l:(l + w), c:(c + h)].shape
                pattern_image[l:(l + w), c:(c + h)] = pattern[:xx, :yy]
    return pattern_image


def get_mask(IMAGE, pattern):
    num_rows, num_cols = IMAGE.shape[:2]
    num_bands = len(np.unique(pattern))
    pattern_image = create_pattern_image(pattern, num_rows, num_cols)
    mask = np.zeros((num_rows, num_cols, num_bands), dtype=bool)
    for k in range(num_bands):
        mask[:, :, k] = (pattern_image == k)
    return mask

def wb_demosaicing(raw, mask, pattern):
    num_rows, num_cols = raw.shape
    num_bands = len(np.unique(pattern))

    H = get_H(pattern)
    IMAGE_WB = np.empty((num_rows, num_cols, num_bands))
    for k in range(num_bands):
        IMAGE_WB[:, :, k] = convolve(raw * mask[:, :, k], H)

    return IMAGE_WB

def create_sparse_image(mosaic_image,pattern_image):
    K_sparse_image = []
    bands = np.unique(pattern_image)
    for k in bands:
        tmp = mosaic_image * (pattern_image==k).astype(bool)
        K_sparse_image.append(tmp)
    K_sparse_image = np.dstack(K_sparse_image)
    return K_sparse_image

def sd_demosaicing(raw, pattern,IMAGE_WB=None,dtype='uint8'):
    raw = raw.astype(float)
    H = get_H(pattern)
    mask = get_mask(raw,pattern)
    num_rows, num_cols = raw.shape
    num_bands = len(np.unique(pattern))
    IMAGE_SD = np.zeros((num_rows, num_cols, num_bands)).astype(float)
    if (IMAGE_WB is None):
        print('Computing IMAGE_WB...')
        IMAGE_WB = wb_demosaicing(raw, mask, pattern)
    for k in range(num_bands):
        for j in range(num_bands):
                diff = convolve((raw-IMAGE_WB[:, :, j]) * mask[:, :, k], H)
                IMAGE_SD[:, :, k] += (raw + diff) * mask[:, :, j]
    IMAGE_SD = np.maximum(IMAGE_SD, 0) # clip to avoid negative values
    if dtype == 'uint8':
        IMAGE_SD = to8Bit(IMAGE_SD)
    elif dtype == 'float32':
        IMAGE_SD = toFloat32(IMAGE_SD)
    else:
        return IMAGE_SD

    return IMAGE_SD
















