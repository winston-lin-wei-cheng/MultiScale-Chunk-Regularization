#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wei-Cheng (Winston) Lin

Script to prepare feature normalization (z-norm) parameters. 
"""
import os
import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat



# loading file table
label_table = pd.read_csv('./MSP-PODCAST-Publish-1.10/Labels/labels_consensus.csv')

# setup audio-text pre-extracted feature root
data_root_audio = './MSP-PODCAST-Publish-1.10/Features/Wav2Vec1024/feat_mat/'
data_root_text = './MSP-PODCAST-Publish-1.10/Features/RoBERTa768/feat_mat/'

# obtain needed table info
whole_fnames = (label_table['FileName'].values).astype('str')
split_set = (label_table['Split_Set'].values).astype('str')
emo_act = label_table['EmoAct'].values
emo_dom = label_table['EmoDom'].values
emo_val = label_table['EmoVal'].values

# obtain features
TrainDataAudio, TrainDataText = [], []
TrainLabel_act, TrainLabel_dom, TrainLabel_val = [], [], []
for i in range(len(whole_fnames)):
    # calculate mean & std per training utterance
    if split_set[i]=='Train':
        dataA = loadmat(data_root_audio + whole_fnames[i].replace('.wav','.mat'))['Audio_data']
        dataT = loadmat(data_root_text + whole_fnames[i].replace('.wav','.mat'))['Text_data']
        TrainDataAudio.append(np.mean(dataA,axis=0))
        TrainDataText.append(np.mean(dataT,axis=0))
        TrainLabel_act.append(emo_act[i])
        TrainLabel_dom.append(emo_dom[i])
        TrainLabel_val.append(emo_val[i])
TrainDataAudio = np.array(TrainDataAudio)
TrainDataText = np.array(TrainDataText)
TrainLabel_act = np.array(TrainLabel_act)
TrainLabel_dom = np.array(TrainLabel_dom)
TrainLabel_val = np.array(TrainLabel_val)

# creating saving repo
if not os.path.isdir('./NormTerm_Speech/'):
    os.makedirs('./NormTerm_Speech/')
if not os.path.isdir('./NormTerm_Text/'):
    os.makedirs('./NormTerm_Text/')

# saving norm parameters (i.e., mean and std of feats based on the 'Train' set)
eps = 1e-10 # assign small eps to prevent std=0 case
FeatAudio_mean_All = np.mean(TrainDataAudio,axis=0)
FeatAudio_std_All = np.std(TrainDataAudio,axis=0)
FeatAudio_std_All[np.where(FeatAudio_std_All==0)[0]] = eps
savemat('./NormTerm_Speech/feat_norm_means.mat', {'normal_para':FeatAudio_mean_All})
savemat('./NormTerm_Speech/feat_norm_stds.mat', {'normal_para':FeatAudio_std_All})
FeatText_mean_All = np.mean(TrainDataText,axis=0)
FeatText_std_All = np.std(TrainDataText,axis=0)
FeatText_std_All[np.where(FeatText_std_All==0)[0]] = eps
savemat('./NormTerm_Text/feat_norm_means.mat', {'normal_para':FeatText_mean_All})
savemat('./NormTerm_Text/feat_norm_stds.mat', {'normal_para':FeatText_std_All})
Label_mean_Act = np.mean(TrainLabel_act)
Label_std_Act = np.std(TrainLabel_act)
savemat('./NormTerm_Speech/act_norm_means.mat', {'normal_para':Label_mean_Act})
savemat('./NormTerm_Speech/act_norm_stds.mat', {'normal_para':Label_std_Act})
savemat('./NormTerm_Text/act_norm_means.mat', {'normal_para':Label_mean_Act})
savemat('./NormTerm_Text/act_norm_stds.mat', {'normal_para':Label_std_Act})
Label_mean_Dom = np.mean(TrainLabel_dom)
Label_std_Dom = np.std(TrainLabel_dom)
savemat('./NormTerm_Speech/dom_norm_means.mat', {'normal_para':Label_mean_Dom})
savemat('./NormTerm_Speech/dom_norm_stds.mat', {'normal_para':Label_std_Dom})
savemat('./NormTerm_Text/dom_norm_means.mat', {'normal_para':Label_mean_Dom})
savemat('./NormTerm_Text/dom_norm_stds.mat', {'normal_para':Label_std_Dom})
Label_mean_Val = np.mean(TrainLabel_val)
Label_std_Val = np.std(TrainLabel_val)
savemat('./NormTerm_Speech/val_norm_means.mat', {'normal_para':Label_mean_Val})
savemat('./NormTerm_Speech/val_norm_stds.mat', {'normal_para':Label_std_Val})
savemat('./NormTerm_Text/val_norm_means.mat', {'normal_para':Label_mean_Val})
savemat('./NormTerm_Text/val_norm_stds.mat', {'normal_para':Label_std_Val})
    