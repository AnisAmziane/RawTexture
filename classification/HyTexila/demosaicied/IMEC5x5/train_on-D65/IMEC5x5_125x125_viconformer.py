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
from descriptors import DNN_models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from types import SimpleNamespace
import numpy as np
import spectral.io.envi as envi
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from importlib import reload
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
training.set_determenistic_mode(random_seed=42, deterministic_cudnn=True)

#*******************************************************
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #
CONFIG = SimpleNamespace(images_path = '/home/anis/Documents/ms_base/others/HyTexila_bmvc/HyTexila_D65_IMEC25_demosaiced/',
                         model_save_path='/TextureExp/classification/exps/trained_models_old/',
                         model_name='Vision_Conformer', input_channels=25, patch_size=(125,125), token_with=5, logits_size=112, device=device)
#
#-----------------------------------------------------------------------------------------#
MSFA_pattern = mt.make_pattern(pattern_type=CONFIG.MSFA)
imageLists = training.get_hdr_and_raw_paths(CONFIG.images_path)
imageLists.sort()
classes = len(imageLists) # each image is a different class
labels = [i for i in range(len(imageLists))] # each image is a different class
#-----------------------------------------------------------------------------------------#
list_train_images = []
list_train_labels = []
#-----------------------------------------------------------------------------------------#
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
    augmented_test_demosaiced = mt.augment_train_raw2_demosaiced(test_raw,MSFA_pattern.shape)
    #Test patch extraction
    for p in range(len(augmented_test_demosaiced)):
        current_test_augmented = augmented_test_demosaiced[p] # being in uint8 makes a big difference
        augmented_train_raw_patches,_ = mt.extract_patches(current_test_augmented, size=CONFIG.patch_size)
        this_train_patch_label = np.asarray([labels[i] for ll in range(len(augmented_train_raw_patches))])
        list_train_images.extend(augmented_train_raw_patches)
        list_train_labels.extend(this_train_patch_label)



trainx, valx, trainy, valy = train_test_split(list_train_images, list_train_labels, train_size=0.95,stratify=list_train_labels, random_state=1)
train_dataset = mt.MakeDataset(trainx,trainy,(CONFIG.patch_size[0],CONFIG.patch_size[0],CONFIG.input_channels))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=False)
#
# val_dataset = mt.MakeDataset(trainx,trainy,(CONFIG.patch_size[0],CONFIG.patch_size[0],CONFIG.input_channels))
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=128,shuffle=False)
#
del train_dataset,valx,trainx
gc.collect()
torch.cuda.empty_cache()

# ------------------- Train Pytorch RawConvMixer ------------------------------------------------------
model = DNN_models.get_model(CONFIG)
print(sum(p.numel() for p in model.parameters()))
model.to(CONFIG.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-4,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
model = training.train(model,CONFIG.model_name,40,optimizer,criterion,train_loader,valdataloader=None,scheduler = scheduler,device=CONFIG.device,
                       save_path=CONFIG.model_save_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1]))


### Extract train features
extracted_features_path = '/TextureExp/classification/exps/features_old/'
descriptor = model.eval().descriptor
cnn_train_features = training.torch_feature_extraction(descriptor, train_loader, device=device)
cnn_train_features = np.vstack(cnn_train_features)

np.save(extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_features.npy',cnn_train_features)
np.save(extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_labels.npy',trainy)









