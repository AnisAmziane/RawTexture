# -*- coding: utf-8 -*-
import sys,gc
sys.path.append('/../')
from utils import mosaic_tools as mt
from utils import SpectralConstancy as sc
from utils import training
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from types import SimpleNamespace
import spectral.io.envi as envi
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.utils.data import TensorDataset
import torch
training.set_determenistic_mode(random_seed=42, deterministic_cudnn=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #
CONFIG = SimpleNamespace(images_path = '/path to HyTexila_A_IMEC25_raw or HyTexila_Solar_IMEC25_raw/',
                         extracted_features_path='/features/',
                         model_save_path='/trained_models/',
                         model_name='RawMixerRes', input_channels=1, patch_size=(125,125),
                         MSFA='Imec25', msfa_width=5, logits_size=112, device=device)
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
        current_test_augmented = augmented_test_raw[p] # being in uint8 makes a big difference
        augmented_test_raw_patches,_ = mt.extract_raw_patches(current_test_augmented, size=CONFIG.patch_size)
        this_test_patch_label = np.asarray([labels[i] for ll in range(len(augmented_test_raw_patches))])
        list_test_images.extend(augmented_test_raw_patches)
        list_test_labels.extend(this_test_patch_label)

#--------------------------------------- Test --------------------------------------------#
list_test_images = np.asarray(list_test_images)
list_test_labels = np.asarray(list_test_labels)
list_test_images = list_test_images.reshape(len(list_test_images),CONFIG.input_channels, CONFIG.patch_size[0], CONFIG.patch_size[1])
# Make it suitable for pytorch
test_dataset = mt.MakeDataset(list_test_images,list_test_labels,(CONFIG.patch_size[0], CONFIG.patch_size[1],CONFIG.input_channels))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=128,shuffle=False)
# load model
model = torch.load(CONFIG.model_save_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'.pth')
descriptor = model.eval().descriptor
# Load train features
cnn_train_features = np.load(CONFIG.extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_features.npy')
cnn_train_labels = np.load(CONFIG.extracted_features_path+CONFIG.model_name+'_'+CONFIG.MSFA+'_'+str(CONFIG.patch_size[0])+'x'+str(CONFIG.patch_size[1])+'_labels.npy')
cnn_test_features = training.torch_feature_extraction(descriptor, test_loader, device=device)
cnn_test_features = np.vstack(cnn_test_features)
gc.collect()
torch.cuda.empty_cache()
## Classify
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', weights='uniform')
knn.fit(cnn_train_features, cnn_train_labels) # labels [0, 111]
# predict
predictions = knn.predict(cnn_test_features)
accuracy = accuracy_score(list_test_labels, predictions)
print('Accuracy : '+str(round(accuracy*100,2))+'%')

