# -*- coding: utf-8 -*-
"""
Weighted Bilinear interpolation method
"""
from scipy.ndimage import convolve
import numpy as np

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

def wb_demosaicing(raw_img,pattern):
    NB_bands = len(np.unique(pattern))
    NB_rows = raw_img.shape[0]
    NB_cols = raw_img.shape[1]
    H = get_H(pattern)
    IMAGE_WB = np.empty((NB_rows, NB_cols,NB_bands))
    pattern_img = create_pattern_image(pattern, NB_rows, NB_cols)
    for k in range(NB_bands):
        IMAGE_WB[:, :,k] = convolve(raw_img[:, :,k] * (pattern_img == k).astype(bool), H)
    return IMAGE_WB

def pii_from_raw(raw_img,pattern):
    H = get_H(pattern)
    ppi = convolve(raw_img, H)
    return ppi

def cube_2_ppi(cube):
    return np.mean(cube,axis=2)

