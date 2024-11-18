# -*- coding: utf-8 -*-
import sys
sys.path.append('/../')
from utils import mosaic_tools as mt
from utils import SpectralConstancy as sc
from utils import training
from descriptors import lbp
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from types import SimpleNamespace
import numpy as np
import spectral.io.envi as envi
from tqdm import tqdm

CONFIG = SimpleNamespace(images_path = '/path to Spectex_D65_IMEC4_raw/',
                         model_save_path='/../',
                            extracted_features_path = '',
                         model_name='MLBP', input_channels=1, patch_size=(124,124),
                         MSFA='Imec4', logits_size=112, device='cpu')

MSFA_pattern = mt.make_pattern(pattern_type=CONFIG.MSFA)
imageLists = training.get_hdr_and_raw_paths(CONFIG.images_path)
imageLists.sort()
classes = len(imageLists) # each image is a different class
patch_size = CONFIG.patch_size
list_train_images = []
list_train_labels = []
labels = [i for i in range(len(imageLists))] # each image is a different class
# -------------------------------------------
for i in tqdm(range(len(imageLists))):
    img_path = imageLists[i]
    obj = envi.open(img_path[0], img_path[1])
    raw = obj.asarray()[:,:,0]
    raw,_ = sc.max_spectral(raw,MSFA_pattern.shape[0])
    raw = sc.normalize8(raw)
    train_raw = raw[:320,:]
    augmented_train_raw = mt.augment_train_raw(train_raw,MSFA_pattern.shape)
    #Train patch extraction
    for p in range(len(augmented_train_raw)):
        current_train_augmented = augmented_train_raw[p] # being in uint8 makes a big difference
        augmented_train_raw_patches,_ = mt.extract_raw_patches(current_train_augmented, size=CONFIG.patch_size)
        this_train_patch_label = np.asarray([labels[i] for ll in range(len(augmented_train_raw_patches))])
        list_train_images.extend(augmented_train_raw_patches)
        list_train_labels.extend(this_train_patch_label)

### Extract train features
trainx, valx, trainy, valy = train_test_split(list_train_images, list_train_labels, train_size=0.95,stratify=list_train_labels, random_state=1)
# trainx, trainy = shuffle(list_train_images, list_train_labels, random_state=1)
trainx = np.asarray(trainx)
trainy = np.asarray(trainy)
trainx = trainx.reshape(-1, trainx.shape[1], trainx.shape[2])
pattern_image = mt.create_pattern_image(MSFA_pattern, CONFIG.patch_size[0], CONFIG.patch_size[1])
executor = Parallel(n_jobs=12, backend='multiprocessing')
B_codes = executor(delayed(lbp.MLBP_codes)(trainx[i], pattern_image, 1) for i in range(len(trainx)))
list_train_features = executor(delayed(lbp.concatenate_hists)(B_codes[i]) for i in range(len(B_codes)))
list_train_features = np.asarray(list_train_features)
### save train features
np.save(CONFIG.extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_features_spectex.npy',list_train_features)
np.save(CONFIG.extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_labels_spectex.npy',trainy)




