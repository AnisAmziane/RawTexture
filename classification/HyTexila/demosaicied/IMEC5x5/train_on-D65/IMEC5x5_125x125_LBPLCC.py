# -*- coding: utf-8 -*-
""" Title: HyTexila texture classification using MSFA-Net features extracted from 200x200 raw patches simulated by IMEC5x5
    Author: Anis Amziane <anisamziane6810@gmail.com>
    Created: 10-Nov-2022
  """
import sys,gc
sys.path.append('/home/anis/PycharmProjects/RawTexture/venv/TextureExp/')
from classification.utils import mosaic_tools as mt
from classification.utils import SpectralConstancy as sc
from classification.utils import training
from descriptors import lbp
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
import numpy as np
import spectral.io.envi as envi
from tqdm import tqdm
import math

#*******************************************************
CONFIG = SimpleNamespace(images_path = '/home/anis/Documents/ms_base/others/HyTexila_bmvc/HyTexila_D65_IMEC25_demosaiced/',
                         model_name='LBPLCC', input_channels=25, patch_size=(65,65), MSFA='Imec25', logits_size=112, device='cpu')
# -----------------------------------------
MSFA_pattern = mt.make_pattern(pattern_type=CONFIG.MSFA)
imageLists = training.get_hdr_and_raw_paths(CONFIG.images_path)
imageLists.sort()
classes = len(imageLists) # each image is a different class
# -----------------------------------------
patch_size = CONFIG.patch_size
labels = [i for i in range(len(imageLists))] # each image is a different class
list_train_images = []
list_train_labels = []
#-------------------------------------------
for i in tqdm(range(len(imageLists))):
    train_images = []
    test_images = []
    img_path = imageLists[i]
    obj = envi.open(img_path[0], img_path[1])
    cube = obj.asarray()
    # raw = sc.to_float32(raw)
    raw = raw[:1020,:1020,0] # 1020 is divisible by 5, 4, and 2
    cube, _ = sc.max_spectral_fully_defined(cube)
    cube = sc.normalize8(cube)
    test_raw = cube[:510,:]
    augmented_test_demosaiced = mt.augment_train_demosaiced(test_raw,MSFA_pattern.shape)
    #Test patch extraction
    for p in range(len(augmented_test_demosaiced)):
        current_train_augmented = augmented_test_demosaiced[p] # being in uint8 makes a big difference
        augmented_train_raw_patches,_ = mt.extract_patches(current_train_augmented, size=CONFIG.patch_size)
        this_train_patch_label = np.asarray([labels[i] for ll in range(len(augmented_train_raw_patches))])
        list_train_images.extend(augmented_train_raw_patches)
        list_train_labels.extend(this_train_patch_label)


### Extract train features
trainx, valx, trainy, valy = train_test_split(list_train_images, list_train_labels, train_size=0.95,stratify=list_train_labels, random_state=1)
#
trainy = np.asarray(trainy)
executor = Parallel(n_jobs=12, backend='multiprocessing')
list_train_features = []
batch_size = 500
steps = int(math.ceil(len(trainx) / batch_size))
i = 0
# memory efficient feature extraction
for j in tqdm(range(steps)):
    data = np.asarray(trainx[i * batch_size:(i + 1) * batch_size])
    list_train_features_ppi = executor(delayed(lbp.simple_lbp_ppi_hist)(data[i],1) for i in range(len(data)))
    list_train_features_ppi = np.asarray(list_train_features_ppi)
    list_train_features_lcc = executor(delayed(lbp.lcc)(data[i], 8, 1,reference='avg',interpolate=False)for i in range(len(data)))
    list_train_features_lcc = executor(delayed(lbp.np_histogram)(list_train_features_lcc[i],256,normalize=False)for i in range(len(list_train_features_lcc)))
    list_train_features_lcc = np.asarray(list_train_features_lcc)
    batch_features = np.concatenate((list_train_features_ppi,list_train_features_lcc),axis=1)
    # free some memory
    del data,trainx[i * batch_size:(i + 1) * batch_size],list_train_features_lcc,list_train_features_ppi
    gc.collect()
    list_train_features.append(batch_features)
    # free some memory
    del batch_features
    gc.collect()
list_train_features = np.vstack(list_train_features)

### save train features
extracted_features_path = '/home/anis/PycharmProjects/RawTexture/venv/TextureExp/classification/exps/features/'
np.save(extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_features.npy',list_train_features)
np.save(extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_labels.npy',trainy)




