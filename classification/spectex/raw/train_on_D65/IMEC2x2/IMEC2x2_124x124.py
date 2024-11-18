# -*- coding: utf-8 -*-
""" Title: Spectex texture classification with IMEC2x2
    Author: Anis Amziane <anisamziane6810@gmail.com>
    Created: 1-Nov-2024
  """
import os
import sys,gc
sys.path.append('/../')
from utils import mosaic_tools as mt
from utils import SpectralConstancy as sc
from utils import training
from descriptors import DNN_models
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
import numpy as np
import spectral.io.envi as envi
from tqdm import tqdm
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
training.set_determenistic_mode(random_seed=42, deterministic_cudnn=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #
CONFIG = SimpleNamespace(images_path = '/path to Spectex_D65_IMEC4_raw/',
                         extracted_features_path = '//',
                         model_save_path='/trained_models/',
                         model_name='VGG11', input_channels=1, patch_size=(124,124), MSFA='Imec4',
                         msfa_width=2, logits_size=60, device=device)
#
MSFA_pattern = mt.make_pattern(pattern_type=CONFIG.MSFA)
imageLists = training.get_hdr_and_raw_paths(CONFIG.images_path)
imageLists.sort()
classes = len(imageLists) # each image is a different class
patch_size = CONFIG.patch_size
labels = [i for i in range(len(imageLists))] # each image is a different class
list_train_images = []
list_train_labels = []
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

trainx, valx, trainy, valy = train_test_split(list_train_images, list_train_labels, train_size=0.95,stratify=list_train_labels, random_state=1)
train_dataset = mt.MakeDataset(trainx,trainy,(CONFIG.patch_size[0],CONFIG.patch_size[0],1))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=128,shuffle=False)
#
val_dataset = mt.MakeDataset(trainx,trainy,(CONFIG.patch_size[0],CONFIG.patch_size[0],1))
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=128,shuffle=False)
#
del val_dataset,train_dataset,valx,trainx
gc.collect()
torch.cuda.empty_cache()

# ------------------- Train Pytorch RawConvMixer ------------------------------------------------------
model = DNN_models.get_model(CONFIG)
print(sum(p.numel() for p in model.parameters()))
model.to(CONFIG.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-4,weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
training.train(model,CONFIG.model_name,30,optimizer,criterion,train_loader,valdataloader=None,scheduler = None,device=CONFIG.device,
                       save_path=CONFIG.model_save_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_spectex')

### Extract train features
descriptor = model.eval().descriptor
#
del model
gc.collect()
torch.cuda.empty_cache()
#
cnn_train_features = training.torch_feature_extraction(descriptor, train_loader, device=device)
cnn_train_features = np.vstack(cnn_train_features)
np.save(CONFIG.extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_spectex_features.npy',cnn_train_features)
np.save(CONFIG.extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_spectex_labels.npy',trainy)
#
del descriptor
gc.collect()
torch.cuda.empty_cache()
