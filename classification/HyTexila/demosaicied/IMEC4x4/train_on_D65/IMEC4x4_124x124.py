# -*- coding: utf-8 -*-
""" Title: HyTexila texture classification using MSFA-Net features extracted from 200x200 raw patches simulated by IMEC5x5
    Author: Anis Amziane <anisamziane6810@gmail.com>
    Created: 10-Nov-2022
  """
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
from importlib import reload
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
training.set_determenistic_mode(random_seed=42, deterministic_cudnn=True)

#*******************************************************
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #
CONFIG = SimpleNamespace(images_path = '/home/anis/Documents/ms_base/others/HyTexila_bmvc/HyTexila_D65_IMEC16_demosaiced/',
                         model_save_path='/home/anis/PycharmProjects/RawTexture/venv/TextureExp/classification/exps/trained_models/',
                         model_name='Vision_Conformer', input_channels=16, patch_size=(124,124),MSFA='Imec16',msfa_width=4, token_width=31, logits_size=112, device=device)
#-----------------------------------------------------------------------------------------#
MSFA_pattern = mt.make_pattern(pattern_type=CONFIG.MSFA)
imageLists = training.get_hdr_and_raw_paths(CONFIG.images_path)
imageLists.sort()
classes = len(imageLists) # each image is a different class
labels = [i for i in range(len(imageLists))] # each image is a different class
list_train_images = []
list_train_labels = []
#-----------------------------------------------------------------------------------------#
for i in tqdm(range(len(imageLists))):
    train_images = []
    test_images = []
    img_path = imageLists[i]
    obj = envi.open(img_path[0], img_path[1])
    cube = obj.asarray()
    cube = cube[:1020, :1020]
    cube, _ = sc.max_spectral_fully_defined(cube)
    cube = sc.normalize8(cube)
    test_raw = cube[:508,:]
    augmented_train_demosaiced = mt.augment_train_demosaiced(test_raw,MSFA_pattern.shape)
    #Test patch extraction
    for p in range(len(augmented_train_demosaiced)):
        current_train_augmented = augmented_train_demosaiced[p] # being in uint8 makes a big difference
        augmented_train_raw_patches,_ = mt.extract_patches(current_train_augmented, size=CONFIG.patch_size)
        this_train_patch_label = np.asarray([labels[i] for ll in range(len(augmented_train_raw_patches))])
        list_train_images.extend(augmented_train_raw_patches)
        list_train_labels.extend(this_train_patch_label)



trainx, valx, trainy, valy = train_test_split(list_train_images, list_train_labels, train_size=0.95,stratify=list_train_labels, random_state=1)
train_dataset = mt.MakeDataset(trainx,trainy,(CONFIG.patch_size[0],CONFIG.patch_size[0],CONFIG.input_channels))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=92,shuffle=False)
#
# val_dataset = mt.MakeDataset(trainx,trainy,(CONFIG.patch_size[0],CONFIG.patch_size[0],CONFIG.input_channels))
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=128,shuffle=False)
#
del train_dataset,valx,trainx
gc.collect()
torch.cuda.empty_cache()

# ------------------- Train Pytorch RawConvMixer ------------------------------------------------------
for model_name in ['ViT','Vision_Conformer','SpectralFormer_CAF','ResNet18','hsi_mixer']:
    gc.collect()
    torch.cuda.empty_cache()
    CONFIG.model_name = model_name
    model = DNN_models.get_model(CONFIG)
    print(sum(p.numel() for p in model.parameters()))
    model.to(CONFIG.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4,weight_decay=1e-5)
    training.train(model,CONFIG.model_name,30,optimizer,criterion,train_loader,valdataloader=None,scheduler = None,device=CONFIG.device,
                           save_path=CONFIG.model_save_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1]))

    ### Extract train features
    extracted_features_path = '/home/anis/PycharmProjects/RawTexture/venv/TextureExp/classification/exps/features/'
    descriptor = model.eval().descriptor
    cnn_train_features = training.torch_feature_extraction(descriptor, train_loader, device=device)
    cnn_train_features = np.vstack(cnn_train_features)

    np.save(extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_features.npy',cnn_train_features)
    np.save(extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_labels.npy',trainy)

    del model,descriptor
    gc.collect()
    torch.cuda.empty_cache()





