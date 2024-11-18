# -*- coding: utf-8 -*-
""" Title: Simulate raw images from K-dimensional MS image based on a specific MSFA pattern
    Author: Anis Amziane <anisamziane6810@gmail.com>
    Created: 10-Nov-2024
  """
import glob
import numpy as np
import spectral.io.envi as envi
import os, sys
sys.path.append('/../')
from utils import mosaic_tools as mt
from tqdm import tqdm

def get_hdr_and_raw_paths(images_path):
    contain_hdrs = glob.iglob(images_path + '**/*.hdr', recursive=True)
    contain_raws = glob.iglob(images_path + '**/*.raw', recursive=True)
    hdrs = []
    raws = []
    imageLists = []
    for f1 in contain_hdrs:
        hdrs.append(f1)
    for f2 in contain_raws:
        raws.append(f2)
    for i in range(len(hdrs)):
        filename = os.path.basename(hdrs[i])
        name, extension = os.path.splitext(filename)
        for ii in range(len(raws)):
            filename2 = os.path.basename(raws[ii])
            name2, extension2 = os.path.splitext(filename2)
            if name == name2:
                temp = (hdrs[i], raws[ii])
                imageLists.append(temp)
    return imageLists

images_path = '/path to dataset/'
target_path = '/path to target folder/'
imageLists = get_hdr_and_raw_paths(images_path)
MSFA_pattern = mt.make_pattern(pattern_type='Simple25')
# ------------------------------------------------------------------------------------ #
for i in tqdm(range(len(imageLists))):
    img_path = imageLists[i]
    obj = envi.open(img_path[0], img_path[1])
    directory = os.path.dirname(obj.filename)  # to check if patch_coordinates.txt exists
    filename = os.path.basename(obj.filename)
    name, extension = os.path.splitext(filename)
    if not os.path.exists(target_path + name):
        os.mkdir(target_path + name)
    folder_path = target_path + name + '/'
    output_path_final = folder_path + name + '.hdr'
    cube = obj.asarray()
    raw = mt.get_mosaic(cube, cube.shape[:2], MSFA_pattern)
    metadata = obj.metadata
    envi.save_image(output_path_final, raw, dtype=np.uint8,interleave='bsq', ext='.raw', metadata=metadata, force=True)



