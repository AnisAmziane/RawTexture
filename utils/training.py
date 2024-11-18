# -*- coding: utf-8 -*-
""" Title: Implementation of different CNN models using PyTorch
    Author: Anis Amziane <anisamziane6810@gmail.com>
    Created: 10-Nov-2022
  """
import math,glob,os
import numpy as np
import random
from tqdm import tqdm
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_determenistic_mode(random_seed=42, deterministic_cudnn=True):
    # can use any of your favorite number for random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    # torch.backends.cudnn.benchmark = True

    if deterministic_cudnn:
       torch.backends.cudnn.deterministic = True # Causes cuDNN to use a deterministic convolution algorithm,
                                              # but may slow down performance.
                                              # It will not guarantee that your training process is deterministic
                                              # if you are using other libraries that may use nondeterministic algorithms


#------------------ Functions ------------------------
def torch_feature_extraction_notarget(feature_extractor, listsamples, batch_size,device='cpu'):
# TODO : fix me using metrics()
    features = []
    steps = int(math.ceil(len(listsamples) / batch_size))
    for i in tqdm(range(steps)):
            data = np.asarray(listsamples[i * batch_size:(i + 1) * batch_size])
            data = torch.tensor(data).to(torch.float32).permute(0,3,1,2)
            with torch.no_grad():
                # Load the data into the GPU if required
                data = data.to(device)
                output = feature_extractor(data)
                features.append(output.cpu().data.numpy())
    return features

def torch_feature_extraction(feature_extractor, dataloader, device='cpu'):
# TODO : fix me using metrics()
    features = []
    for batch_idx, (data, target) in tqdm(enumerate(dataloader),total=len(dataloader)):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            output = feature_extractor(data)
            features.append(output.cpu().data.numpy())
    return features

def val(net, data_loader, device='cpu'):
# TODO : fix me using metrics()
    predictions = []
    for batch_idx, (data, target) in tqdm(enumerate(data_loader),total=len(data_loader)):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            output = net(data)
            output = torch.log_softmax(output, dim=1)
            output = output.cpu().data.numpy().argmax(axis=1)
            # _, output = torch.max(output, dim=1)
            predictions.extend(output)
    return predictions

def get_accuracy(y_pred, y_test):
    # y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    # _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = y_pred == y_test
    acc = correct_pred.sum() / len(correct_pred)
    acc = np.round(acc * 100)
    return acc

def perbatch_prediction(samples,model,batch_size):
    import math
    predictions = []
    steps = int(math.ceil(len(samples)/batch_size))
    for i in range(steps):
       # predictions.append(model.predict(samples[i*batch_size:(i+1)*batch_size],batch_size=64))
       this_batch = np.asarray(samples[i * batch_size:(i + 1) * batch_size])
       predictions.append(model.predict(this_batch))
    return np.hstack(predictions)

def torch_perbatch_prediction(samples,model,batch_size,device):
    import math
    predictions = []
    steps = int(math.ceil(len(samples)/batch_size))
    for i in tqdm(range(steps)):
       this_batch = torch.from_numpy(np.asarray(samples[i * batch_size:(i + 1) * batch_size])).permute(0,3,1,2).to(device)
       predictions.append(F.softmax(model(this_batch),1).detach().cpu().numpy())
    return predictions

def perbatch_prediction(samples,model,mini_batch,dims=(3,64,64)):
    predictions = []
    steps = int(math.ceil(len(samples)/mini_batch))
    for i in range(steps):
       this_batch = samples[i * mini_batch:(i + 1) * mini_batch]
       this_batch = this_batch.reshape((len(this_batch), dims[0], dims[1],dims[2]))
       predictions.append(model(this_batch))
    return torch.vstack(predictions)




def train(model,epochs,optimizer,criterion,dataloader,valdataloader=None,scheduler=None,device='cuda',save_path=None):
    set_determenistic_mode(random_seed=42, deterministic_cudnn=True)
    device = device
    losses = np.zeros(1000000)
    best_val_accuracy = 0
    iter_ = 1
    for e in tqdm(range(1, epochs + 1), desc="Training the network"):
        # Set the network to training mode
        model.train()
        epoch_loss = 0.
        all_train_targets = []
        all_train_predictions = []
        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Load the data into the GPU if required
            data, target = data.to(device).float(), target.to(device)
            all_train_targets.extend(target.cpu().data.numpy())
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses[iter_] = loss.item()
            ## record the average training loss
            epoch_loss += loss.cpu().detach().data
            # ---------------------------Train Loss/Accuracy -----------------------------------------
            output = torch.log_softmax(output, dim=1).argmax(axis=1)
            all_train_predictions.extend(output.cpu().data.numpy())
            del data, target
            gc.collect()
            torch.cuda.empty_cache()
        if scheduler is not None:
            scheduler.step()
        if np.isscalar(all_train_targets[0]):
             train_epoch_accuracy = get_accuracy(torch.tensor(all_train_predictions), torch.tensor(all_train_targets))
        else:
            train_epoch_accuracy = get_accuracy(torch.tensor(all_train_predictions),torch.tensor(all_train_targets).argmax(axis=1))
        print('Train epoch Loss: {:.6f}\t Train epoch Accuracy: {:.6f} '.format(epoch_loss/len(dataloader), train_epoch_accuracy))
            # validate the model
        if valdataloader:
            model.eval()
            print('\n')
            print('Evaluating the model')
            print('\n')
            valid_loss = 0
            # val_accuracy = 0
            all_val_targets = []
            all_val_predictions = []
            for batch_idx, (data, target) in tqdm(enumerate(valdataloader), total=len(valdataloader)):
                # move to GPU
                data, target = data.to(device), target.to(device)
                all_val_targets.extend(target.cpu().data.numpy())
                ## update the average validation loss
                output = model(data)
                loss = criterion(output, target)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                #---------------------------Accuracy-----------------------------------------
                output = torch.log_softmax(output, dim=1).argmax(axis=1)
                all_val_predictions.extend(output.cpu().data.numpy())
            if np.isscalar(all_val_targets[0]):
                val_accuracy = get_accuracy(torch.tensor(all_val_predictions),torch.tensor(all_val_targets))
            else:
                val_accuracy = get_accuracy(torch.tensor(all_val_predictions),torch.tensor(all_val_targets).argmax(axis=1))
            if best_val_accuracy < val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = model

            del data, target
            gc.collect()
            torch.cuda.empty_cache()
            print('Epoch: {} \tValidation Accuracy: {:.6f} \tValidation Loss: {:.6f}'.format(e,val_accuracy,valid_loss/len(valdataloader)))
        else:
            best_model = model

    if save_path is not None:
        torch.save(best_model, save_path + '.pth')
        print('Model saved!')
    # return best_model


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

