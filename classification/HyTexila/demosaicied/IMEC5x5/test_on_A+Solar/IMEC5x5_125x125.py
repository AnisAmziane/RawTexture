# -*- coding: utf-8 -*-
import os,sys, glob,random,gc
sys.path.append('/home/anis/PycharmProjects/RawTexture/venv/TextureExp/')
from classification.utils import mosaic_tools as mt
from classification.utils import SpectralConstancy as sc
from classification.utils import training
from descriptors import DNN_models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from types import SimpleNamespace
import spectral.io.envi as envi
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
from importlib import reload
training.set_determenistic_mode(random_seed=42, deterministic_cudnn=True)


#*****************************************************************************************#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #
CONFIG = SimpleNamespace(images_path = '/home/anis/Documents/ms_base/others/HyTexila_bmvc/HyTexila_Solar_IMEC25_demosaiced/',
                         model_save_path='/home/anis/PycharmProjects/RawTexture/venv/TextureExp/classification/exps/trained_models/',
                         model_name='SpectralFormer_CAF', input_channels=25, patch_size=(125,125),MSFA='Imec25', token_width=25, logits_size=112, device=device)

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
    # raw = sc.to_float32(raw)
    cube = cube[:1020, :1020]
    cube, _ = sc.max_spectral_fully_defined(cube)
    cube = sc.normalize8(cube)
    test_raw = cube[510:,:]
    augmented_test_demosaiced = mt.augment_test_demosaiced(test_raw,MSFA_pattern.shape)
    #Test patch extraction
    for p in range(len(augmented_test_demosaiced)):
        current_test_augmented = augmented_test_demosaiced[p] # being in uint8 makes a big difference
        augmented_test_raw_patches,_ = mt.extract_patches(current_test_augmented, size=CONFIG.patch_size)
        this_test_patch_label = np.asarray([labels[i] for ll in range(len(augmented_test_raw_patches))])
        list_test_images.extend(augmented_test_raw_patches)
        list_test_labels.extend(this_test_patch_label)

#--------------------------------------- Test --------------------------------------------
# load model
model = torch.load(CONFIG.model_save_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'.pth')
descriptor = model.eval().descriptor
#
list_test_images = np.asarray(list_test_images)
list_test_labels = np.asarray(list_test_labels)
# Make it suitable for pytorch
test_dataset = mt.MakeDataset(list_test_images,list_test_labels,(CONFIG.patch_size[0], CONFIG.patch_size[1],CONFIG.input_channels))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=128,shuffle=False)
# Load train features
# Load train features
extracted_features_path = '/home/anis/PycharmProjects/RawTexture/venv/TextureExp/classification/exps/features/'
cnn_train_features = np.load(extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_features.npy')
cnn_train_labels = np.load(extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_labels.npy')
#
cnn_test_features = training.torch_feature_extraction(descriptor, test_loader, device=device)
cnn_test_features = np.vstack(cnn_test_features)
#
gc.collect()
torch.cuda.empty_cache()
## Classify
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', weights='uniform')
knn.fit(cnn_train_features, cnn_train_labels) # labels [0, 111]
# predict
predictions = knn.predict(cnn_test_features)
accuracy = accuracy_score(list_test_labels, predictions)
print('Accuracy : '+str(round(accuracy*100,2))+'%')
#
del model,descriptor
gc.collect()
torch.cuda.empty_cache()


# ### --------------------------------- LBP-based ------------------------------------ ###
# ### 1- M-LBP
# from joblib import Parallel, delayed
# executor = Parallel(n_jobs=12, backend='multiprocessing')
# trainx = list_test_images.reshape(-1, trainx.shape[1], trainx.shape[2])
# MSFA = 'Imec25'
# pattern = mt.make_pattern(pattern_type=MSFA)
# dims = (64,64)
# pattern_image = mt.create_pattern_image(pattern, dims[0], dims[1])
# B_codes = executor(delayed(lbp.MLBP_codes)(trainx[i], pattern_image, 1) for i in range(len(trainx)))
# list_train_features = executor(delayed(lbp.concatenate_hists)(B_codes[i]) for i in range(len(B_codes)))
# ###
# test_images = list_test_images.reshape(-1, list_test_images.shape[2], list_test_images.shape[3])
# B_codes_test = executor(delayed(lbp.MLBP_codes)(test_images[i], pattern_image, 1) for i in range(len(test_images)))
# list_test_features = executor(delayed(lbp.concatenate_hists)(B_codes_test[i]) for i in range(len(B_codes_test)))
# ###
# list_test_features = np.asarray(list_test_features)
# list_train_features = np.asarray(list_train_features)
# ###
# knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', weights='uniform')
# knn.fit(list_train_features, trainy) # labels [0, 111]
# # predict
# predictions = knn.predict(list_test_features)
# accuracy = accuracy_score(list_test_labels, predictions)
# print('Accuracy : '+str(round(accuracy*100,2))+'%')






