#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:40:37 2019

@author: winston
"""
import torch
import sys
import numpy as np
import os
from utils import cc_coef, evaluation_metrics, ParymidTempPool, RdnMultiRslChunk
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader import MspPodcastDataset_SpeechText
from torch.utils.data.sampler import SubsetRandomSampler
from model import TransformerENC_CoAtten



def collate_fn_train(batch):
    # list of batched data/labels
    data_speech, data_text, label_act, label_dom, label_val = zip(*batch)
    
    actual_lens_speech = []
    actual_lens_text = []
    Batch_PadData_Speech = []
    Batch_PadData_Text = []
    for i in range(len(data_speech)):
        
        # temporal pooling operation for chunk-level speech representation
        ds, actualL = RdnMultiRslChunk(data_speech[i], sample_pool=[0.2, 0.4, 0.6, 0.8, 1.0])
        ds = ParymidTempPool(ds, actual_lengths=actualL)
        dt = data_text[i]
        
        # prepend the global sentence-level summary token
        ds = np.concatenate((np.ones((1,ds.shape[-1])), ds),axis=0)
        dt = np.concatenate((np.ones((1,dt.shape[-1])), dt),axis=0)
        
        # padding sequence to the desired max_length & record actual lengths
        actual_lens_speech.append(len(ds))
        actual_lens_text.append(len(dt))
        _pads_sph = np.zeros((max_length-len(ds),ds.shape[-1]))
        _pads_txt = np.zeros((max_length-len(dt),dt.shape[-1]))
        ds = np.concatenate((ds, _pads_sph),axis=0)
        dt = np.concatenate((dt, _pads_txt),axis=0)
        Batch_PadData_Speech.append(ds)
        Batch_PadData_Text.append(dt)
        
    # generating the padding masks
    actual_lens_speech = torch.from_numpy(np.array(actual_lens_speech))
    actual_lens_text = torch.from_numpy(np.array(actual_lens_text))
    pads_mask_sph = torch.arange(max_length).expand(len(actual_lens_speech), max_length) >= actual_lens_speech.unsqueeze(1)
    pads_mask_txt = torch.arange(max_length).expand(len(actual_lens_text), max_length) >= actual_lens_text.unsqueeze(1)
    
    # prepare numpy arrays to Torch Tensor
    Batch_PadData_Speech = np.array(Batch_PadData_Speech)
    Batch_PadData_Text = np.array(Batch_PadData_Text)
    label_act = np.array(label_act)
    label_dom = np.array(label_dom)
    label_val = np.array(label_val)
    
    return torch.from_numpy(Batch_PadData_Speech), torch.from_numpy(Batch_PadData_Text), torch.from_numpy(label_act), torch.from_numpy(label_dom), torch.from_numpy(label_val), pads_mask_sph, pads_mask_txt

def collate_fn_eval(batch):
    # list of batched data/labels
    data_speech, data_text, label_act, label_dom, label_val = zip(*batch)
    
    actual_lens_speech = []
    actual_lens_text = []
    Batch_PadData_Speech = []
    Batch_PadData_Text = []
    for i in range(len(data_speech)):
        
        # temporal pooling operation for chunk-level speech representation
        ds, actualL = RdnMultiRslChunk(data_speech[i], sample_pool=[1.0])
        ds = ParymidTempPool(ds, actual_lengths=actualL)
        dt = data_text[i]
        
        # prepend the global sentence-level summary token
        ds = np.concatenate((np.ones((1,ds.shape[-1])), ds),axis=0)
        dt = np.concatenate((np.ones((1,dt.shape[-1])), dt),axis=0)
        
        # padding sequence to the desired max_length & record actual lengths
        actual_lens_speech.append(len(ds))
        actual_lens_text.append(len(dt))
        _pads_sph = np.zeros((max_length-len(ds),ds.shape[-1]))
        _pads_txt = np.zeros((max_length-len(dt),dt.shape[-1]))
        ds = np.concatenate((ds, _pads_sph),axis=0)
        dt = np.concatenate((dt, _pads_txt),axis=0)
        Batch_PadData_Speech.append(ds)
        Batch_PadData_Text.append(dt)
        
    # generating the padding masks
    actual_lens_speech = torch.from_numpy(np.array(actual_lens_speech))
    actual_lens_text = torch.from_numpy(np.array(actual_lens_text))
    pads_mask_sph = torch.arange(max_length).expand(len(actual_lens_speech), max_length) >= actual_lens_speech.unsqueeze(1)
    pads_mask_txt = torch.arange(max_length).expand(len(actual_lens_text), max_length) >= actual_lens_text.unsqueeze(1)
    
    # prepare numpy arrays to Torch Tensor
    Batch_PadData_Speech = np.array(Batch_PadData_Speech)
    Batch_PadData_Text = np.array(Batch_PadData_Text)
    label_act = np.array(label_act)
    label_dom = np.array(label_dom)
    label_val = np.array(label_val)
    
    return torch.from_numpy(Batch_PadData_Speech), torch.from_numpy(Batch_PadData_Text), torch.from_numpy(label_act), torch.from_numpy(label_dom), torch.from_numpy(label_val), pads_mask_sph, pads_mask_txt

def model_validation(model, valid_loader):
    model.eval()
    with torch.no_grad():
        batch_loss_valid_all = []
        for _, data_batch in enumerate(tqdm(valid_loader, file=sys.stdout)):
            # Input Tensor Data/Labels/Masks
            inp_ds, inp_dt, tar_act, tar_dom, tar_val, msk_ds, msk_dt = data_batch
            inp_ds, inp_dt = inp_ds.cuda().float(), inp_dt.cuda().float()
            tar_act, tar_dom, tar_val = tar_act.cuda().float(), tar_dom.cuda().float(), tar_val.cuda().float()
            msk_ds, msk_dt = msk_ds.cuda(), msk_dt.cuda()
            # models flow
            pred_act, pred_dom, pred_val = model(inp_ds, inp_dt, msk_ds, msk_dt)
            # loss calculation
            loss = (cc_coef(pred_act, tar_act) + cc_coef(pred_dom, tar_dom) + cc_coef(pred_val, tar_val))/3
            batch_loss_valid_all.append(loss.data.cpu().numpy())
            torch.cuda.empty_cache()
    return np.mean(batch_loss_valid_all)

def model_testing(model, test_loader):
    # loading de-norm parameters
    from scipy.io import loadmat
    norm_parameters_speech = 'NormTerm_Speech'
    Label_mean_act = loadmat('../'+norm_parameters_speech+'/act_norm_means.mat')['normal_para'][0][0]
    Label_std_act = loadmat('../'+norm_parameters_speech+'/act_norm_stds.mat')['normal_para'][0][0]
    Label_mean_dom = loadmat('../'+norm_parameters_speech+'/dom_norm_means.mat')['normal_para'][0][0]
    Label_std_dom = loadmat('../'+norm_parameters_speech+'/dom_norm_stds.mat')['normal_para'][0][0]
    Label_mean_val = loadmat('../'+norm_parameters_speech+'/val_norm_means.mat')['normal_para'][0][0]
    Label_std_val = loadmat('../'+norm_parameters_speech+'/val_norm_stds.mat')['normal_para'][0][0]
    # model testing mode
    model.eval()
    Pred_Act = []
    Pred_Dom = []
    Pred_Val = []
    GT_Act = []
    GT_Dom = []
    GT_Val = []
    with torch.no_grad():
        for _, data_batch in enumerate(tqdm(test_loader, file=sys.stdout)):
            # Input Tensor Data/Labels/Masks
            inp_ds, inp_dt, tar_act, tar_dom, tar_val, msk_ds, msk_dt = data_batch
            inp_ds, inp_dt = inp_ds.cuda().float(), inp_dt.cuda().float()
            msk_ds, msk_dt = msk_ds.cuda(), msk_dt.cuda()
            # models flow
            pred_act, pred_dom, pred_val = model(inp_ds, inp_dt, msk_ds, msk_dt)
            # output predictions
            Pred_Act.extend(pred_act.data.cpu().numpy().tolist())
            Pred_Dom.extend(pred_dom.data.cpu().numpy().tolist())
            Pred_Val.extend(pred_val.data.cpu().numpy().tolist())
            GT_Act.extend(tar_act.data.cpu().numpy().tolist())
            GT_Dom.extend(tar_dom.data.cpu().numpy().tolist())
            GT_Val.extend(tar_val.data.cpu().numpy().tolist())
            torch.cuda.empty_cache()
    # de-norm GT and preds
    Pred_Act = (Label_std_act* np.array(Pred_Act)) + Label_mean_act
    Pred_Dom = (Label_std_dom* np.array(Pred_Dom)) + Label_mean_dom
    Pred_Val = (Label_std_val* np.array(Pred_Val)) + Label_mean_val
    GT_Act = (Label_std_act* np.array(GT_Act)) + Label_mean_act
    GT_Dom = (Label_std_dom* np.array(GT_Dom)) + Label_mean_dom
    GT_Val = (Label_std_val* np.array(GT_Val)) + Label_mean_val
    # compute final CCC performance
    pred_Act_CCC = evaluation_metrics(GT_Act, Pred_Act)[0]
    pred_Dom_CCC = evaluation_metrics(GT_Dom, Pred_Dom)[0]
    pred_Val_CCC = evaluation_metrics(GT_Val, Pred_Val)[0]
    return pred_Act_CCC, pred_Dom_CCC, pred_Val_CCC
###############################################################################


# Parameters
iter_max = 5000
batch_size = 128
shuffle = True
max_length = 128

# I/O PATH settings
SAVING_PATH = './Models/'
label_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.10/Labels/labels_consensus.csv'
root_dir = ['/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.10/Features/Wav2Vec1024/feat_mat/',
            '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.10/Features/RoBERTa_Embedding768/feat_mat/']

# loading the model
model = TransformerENC_CoAtten(hidden_dim=256, nhead=4, max_len=max_length)
model.cuda()

# creating saving repo
if not os.path.isdir(SAVING_PATH):
    os.makedirs(SAVING_PATH)

# loading datasets
training_dataset = MspPodcastDataset_SpeechText(root_dir, label_dir, split_set='Train')
validation_dataset = MspPodcastDataset_SpeechText(root_dir, label_dir, split_set='Development')
testing_dataset = MspPodcastDataset_SpeechText(root_dir, label_dir, split_set='Test1')

# shuffle datasets by generating random indices 
train_indices = list(range(len(training_dataset)))
valid_indices = list(range(len(validation_dataset)))
if shuffle:
    np.random.shuffle(train_indices)

# creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
train_loader = torch.utils.data.DataLoader(training_dataset,
                                           batch_size=batch_size,
                                           sampler=train_sampler,
                                           num_workers=0,
                                           pin_memory=True,
                                           collate_fn=collate_fn_train)

valid_sampler = SubsetRandomSampler(valid_indices)
valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                           batch_size=batch_size,
                                           sampler=valid_sampler,
                                           num_workers=0,
                                           pin_memory=True,
                                           collate_fn=collate_fn_eval)

test_loader = torch.utils.data.DataLoader(testing_dataset,
                                          batch_size=batch_size,
                                          num_workers=0,
                                          pin_memory=True,
                                          collate_fn=collate_fn_eval)

# create an optimizer for training
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# emotion-recog model training (Iteration-Based)
Iter_trainLoss_All = []
Iter_validLoss_All = []
val_loss_best = 0
iter_count = 0
num_iter_to_valid = 100
while True:
    # stopping criteria
    if iter_count>=iter_max:
        break    
    
    for _, data_batch in enumerate(tqdm(train_loader, file=sys.stdout)):
        # iter setting & record
        model.train()
        iter_count += 1
        
        # Input Tensor Data/Labels/Masks
        inp_ds, inp_dt, tar_act, tar_dom, tar_val, msk_ds, msk_dt = data_batch
        inp_ds, inp_dt = inp_ds.cuda().float(), inp_dt.cuda().float()
        tar_act, tar_dom, tar_val = tar_act.cuda().float(), tar_dom.cuda().float(), tar_val.cuda().float()
        msk_ds, msk_dt = msk_ds.cuda(), msk_dt.cuda()

        # models flow
        pred_act, pred_dom, pred_val = model(inp_ds, inp_dt, msk_ds, msk_dt)
        
        # MTL-CCC loss
        loss = (cc_coef(pred_act, tar_act) + cc_coef(pred_dom, tar_dom) + cc_coef(pred_val, tar_val))/3
        train_loss = loss.data.cpu().numpy()
        Iter_trainLoss_All.append(train_loss)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # clear GPU memory
        torch.cuda.empty_cache()
        
        # do the model validation every XX iterations
        if iter_count%num_iter_to_valid==0:
            print('validation process')
            val_loss = model_validation(model, valid_loader)
            Iter_validLoss_All.append(val_loss)
            print('Iteration: '+str(iter_count)+' ,Training-loss: '+str(train_loss)+' ,Validation-loss: '+str(val_loss))
            print('=================================================================')
    
            # Checkpoint for saving best model based on val-loss
            if iter_count/num_iter_to_valid==1:
                val_loss_best = val_loss
                torch.save(model.state_dict(), os.path.join(SAVING_PATH, 'TransformerCoAtten_iter'+str(iter_max)+'_batch'+str(batch_size)+'_AudioText_MultiRslParymidPool_MTL.pth.tar'))
                print("=> Saving the initial best model (Iteration="+str(iter_count)+")")
            else:
                if val_loss_best > val_loss:
                    torch.save(model.state_dict(), os.path.join(SAVING_PATH, 'TransformerCoAtten_iter'+str(iter_max)+'_batch'+str(batch_size)+'_AudioText_MultiRslParymidPool_MTL.pth.tar'))
                    print("=> Saving a new best model (Iteration="+str(iter_count)+")")
                    print("=> Loss reduction from "+str(val_loss_best)+" to "+str(val_loss) )
                    val_loss_best = val_loss
                else:
                    print("=> Validation Loss did not improve (Iteration="+str(iter_count)+")")
            print('=================================================================')        

# Drawing Loss Curve for Epoch-based and Batch-based
Iter_trainLoss_All = np.mean(np.array(Iter_trainLoss_All[:len(Iter_validLoss_All)*num_iter_to_valid]).reshape(-1, num_iter_to_valid), axis=1).tolist()
plt.title('Epoch-Loss Curve')
plt.plot(Iter_trainLoss_All,color='blue',linewidth=3)
plt.plot(Iter_validLoss_All,color='red',linewidth=3)
plt.savefig(os.path.join(SAVING_PATH, 'TransformerCoAtten_iter'+str(iter_max)+'_batch'+str(batch_size)+'_AudioText_MultiRslParymidPool_MTL.png'))

# re-loading the best model and do the testing stage
MODEL_PATH = SAVING_PATH+'TransformerCoAtten_iter'+str(iter_max)+'_batch'+str(batch_size)+'_AudioText_MultiRslParymidPool_MTL.pth.tar'
model = TransformerENC_CoAtten(hidden_dim=256, nhead=4, max_len=max_length)
model.load_state_dict(torch.load(MODEL_PATH))
model.cuda()
CCC_Act, CCC_Dom, CCC_Val = model_testing(model, test_loader)
print('#########################################################')
print('## Summary Performance on MSP-PODCAST v1.10 Test1 Set ##')
print('#########################################################')
print('Iterations: '+str(iter_max))
print('Batch_Size: '+str(batch_size))
print('Model_Type: TransformerCoAtten-AudioText-ParymidPool-MultiRslChunk')
print('Act-CCC: '+str(CCC_Act))
print('Dom-CCC: '+str(CCC_Dom))
print('Val-CCC: '+str(CCC_Val))
print('===================================================')

