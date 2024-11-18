# -*- coding: utf-8 -*-
""" Title: Select a K-dimensional MS image from a B-dimensional one (K <<B) based on specific MSFA bands
    Author: Anis Amziane <anisamziane6810@gmail.com>
    Created: 10-Nov-2024
  """

import glob
import numpy as np
import spectral.io.envi as envi
import os, sys
sys.path.append('../')
from utils import params
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
target_path = '/destination folder/'
imageLists = get_hdr_and_raw_paths(images_path)
# ------------------------------------------# ------------------------------------------
HyTexila_centers = params.HyTexila_centers
target_centers = params.IMEC25_sorted_centers
selected_centers_idx = params.selected_IMEC25_centers_idx

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
    rows,cols,K = cube.shape
    selected_cube = cube[:,:,selected_centers_idx] # closest channels to IMEC MSFA bands
    #
    metadata = obj.metadata
    metadata['wavelength'] = target_centers
    envi.save_image(output_path_final, selected_cube, dtype=np.uint8,interleave='bsq', ext='.raw', metadata=metadata, force=True)


