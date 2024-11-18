# -*- coding: utf-8 -*-
""" Title: HyTexila texture classification using MSFA-Net features extracted from 200x200 raw patches simulated by IMEC5x5
    Author: Anis Amziane <anisamziane6810@gmail.com>
    Created: 10-Nov-2022
  """
import sys,gc
sys.path.append('/../')
from classification.utils import mosaic_tools as mt
from classification.utils import SpectralConstancy as sc
from classification.utils import training
from descriptors import lbp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from types import SimpleNamespace
import numpy as np
import spectral.io.envi as envi
from tqdm import tqdm

#-----------------------------------------------------------------------------------------#
CONFIG = SimpleNamespace(images_path = '/path to HyTexila_Solar_IMEC25_Raw/',
                         extracted_features_path='/features/',
                         model_name='MLBP', input_channels=1,
                         patch_size=(125,125), MSFA='Imec25',
                         logits_size=112, device='cpu')
#-----------------------------------------------------------------------------------------#
MSFA_pattern = mt.make_pattern(pattern_type=CONFIG.MSFA)
imageLists = training.get_hdr_and_raw_paths(CONFIG.images_path)
imageLists.sort()
classes = len(imageLists) # each image is a different class
labels = [i for i in range(len(imageLists))] # each image is a different class
list_test_images = []
list_test_labels = []
#-----------------------------------------------------------------------------------------#
for i in tqdm(range(len(imageLists))):
    train_images = []
    test_images = []
    img_path = imageLists[i]
    obj = envi.open(img_path[0], img_path[1])
    raw = obj.asarray()
    raw = raw[:1020,:1020,0] # 1020 is divisible by 5, 4, and 2
    raw, _ = sc.max_spectral(raw, MSFA_pattern.shape[0])
    raw = sc.normalize8(raw)
    test_raw = raw[510:,:]
    augmented_test_raw = mt.augment_test_raw(test_raw,MSFA_pattern.shape)
    #Test patch extraction
    for p in range(len(augmented_test_raw)):
        current_test_augmented = augmented_test_raw[p]
        augmented_test_raw_patches,_ = mt.extract_raw_patches(current_test_augmented, size=CONFIG.patch_size)
        this_test_patch_label = np.asarray([labels[i] for ll in range(len(augmented_test_raw_patches))])
        list_test_images.extend(augmented_test_raw_patches)
        list_test_labels.extend(this_test_patch_label)

### Extract test features
list_test_images = np.asarray(list_test_images)
list_test_labels = np.asarray(list_test_labels)
list_test_images = list_test_images.reshape(-1, list_test_images.shape[1], list_test_images.shape[2])
# extract test features
executor = Parallel(n_jobs=12, backend='multiprocessing')
pattern_image = mt.create_pattern_image(MSFA_pattern, CONFIG.patch_size[0], CONFIG.patch_size[1])
B_codes = executor(delayed(lbp.MLBP_codes)(list_test_images[i], pattern_image, 1) for i in range(len(list_test_images)))
list_test_features = executor(delayed(lbp.concatenate_hists)(B_codes[i]) for i in range(len(B_codes)))
list_test_labels = np.asarray(list_test_labels)
# Load train features
list_train_features = np.load(CONFIG.extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_features.npy')
list_train_labels = np.load(CONFIG.extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_labels.npy')
#
clf = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
clf.fit(list_train_features, list_train_labels) # labels [0, 111]
# predict
predictions = clf.predict(np.asarray(list_test_features))
accuracy = accuracy_score(list_test_labels, predictions)
print('Accuracy : '+str(round(accuracy*100,2))+'%')






