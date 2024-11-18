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
# from sklearn.neighbors import KNeighborsClassifier
from cuml.neighbors import KNeighborsClassifier as cuKNeighbors
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from types import SimpleNamespace
import numpy as np
import spectral.io.envi as envi
from tqdm import tqdm
import math

#*******************************************************
CONFIG = SimpleNamespace(images_path = '/home/anis/Documents/ms_base/others/HyTexila_bmvc/HyTexila_Solar_IMEC25_demosaiced/',
                         model_name='LBP', input_channels=25, patch_size=(125,125), MSFA='Imec25', logits_size=112, device='cpu')
#
#-----------------------------------------------------------------------------------------#
MSFA_pattern = mt.make_pattern(pattern_type=CONFIG.MSFA)
imageLists = training.get_hdr_and_raw_paths(CONFIG.images_path)
imageLists.sort()
classes = len(imageLists) # each image is a different class
labels = [i for i in range(len(imageLists))] # each image is a different class
#-----------------------------------------------------------------------------------------#
list_test_images = []
list_test_labels = []
#-----------------------------------------------------------------------------------------#
for i in tqdm(range(len(imageLists))):
    train_images = []
    test_images = []
    img_path = imageLists[i]
    obj = envi.open(img_path[0], img_path[1])
    cube = obj.asarray()
    # cube = sc.to_float32(cube)
    cube = cube[:1020,:1020]
    cube,_ = sc.max_spectral_fully_defined(cube)
    cube = sc.normalize8(cube)
    test_cube = cube[510:, :]
    augmented_test_demosaiced = mt.augment_test_demosaiced(test_cube,MSFA_pattern.shape)
    #Test patch extraction
    for p in range(len(augmented_test_demosaiced)):
        current_test_augmented = augmented_test_demosaiced[p] # being in uint8 makes a big difference
        augmented_test_demosaiced_patches,_ = mt.extract_patches(current_test_augmented, size=CONFIG.patch_size)
        this_test_patch_label = np.asarray([labels[i] for ll in range(len(augmented_test_demosaiced_patches))])
        list_test_images.extend(augmented_test_demosaiced_patches)
        list_test_labels.extend(this_test_patch_label)
### Extract test features
list_test_labels = np.asarray(list_test_labels)
# extract test features
executor = Parallel(n_jobs=10, backend='multiprocessing')
list_test_features = []
batch_size = 500
steps = int(math.ceil(len(list_test_labels) / batch_size))
# memory efficient feature extraction
for j in tqdm(range(steps)):
        data = np.asarray(list_test_images[j * batch_size:(j + 1) * batch_size])
        batch_features = executor(delayed(lbp.get_concatenated_hists_simple)(data[i], 8, 1) for i in range(len(data)))
        batch_features = np.vstack(batch_features)
        list_test_features.append(batch_features)
        # free some memory
        del batch_features,data
        gc.collect()
list_test_features = np.vstack(list_test_features)

# Load train features
extracted_features_path = '/home/anis/PycharmProjects/RawTexture/venv/TextureExp/classification/exps/features/'
list_train_features = np.load(extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_features.npy')
list_train_labels = np.load(extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_labels.npy')
###
knn = cuKNeighbors(n_neighbors=1)
knn.fit(list_train_features, list_train_labels) # labels [0, 111]
# predict
predictions = knn.predict(list_test_features)
accuracy = accuracy_score(list_test_labels, predictions)
print('Accuracy : '+str(round(accuracy*100,2))+'%')
#
del knn
gc.collect()






