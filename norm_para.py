#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:27:46 2019

@author: winston
"""
import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat



if __name__=='__main__': 
    # Get Label-Table & Data-Feature
    label_table = pd.read_csv('/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.10/Labels/labels_consensus.csv')

    # data_root = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.10/Features/Wav2Vec1024/feat_mat/'
    data_root = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.10/Features/RoBERTa_Embedding768/feat_mat/'
    
    # Get Desired Attributes from Label-Table
    whole_fnames = (label_table['FileName'].values).astype('str')
    split_set = (label_table['Split_Set'].values).astype('str')
    emo_act = label_table['EmoAct'].values
    emo_dom = label_table['EmoDom'].values
    emo_val = label_table['EmoVal'].values

    # Acoustic-Feature Normalization based on Training Set
    Train_Data = []
    Train_Label_act = []
    Train_Label_dom = []
    Train_Label_val = []
    for i in range(len(whole_fnames)):
        # Calculate Mean & Std per training utterance
        if split_set[i]=='Train':
            # data = loadmat(data_root + whole_fnames[i].replace('.wav','.mat'))['Audio_data']
            data = loadmat(data_root + whole_fnames[i].replace('.wav','.mat'))['Text_data']
            Train_Data.append(np.mean(data,axis=0)) # mean first or will out-of-memory
            Train_Label_act.append(emo_act[i])
            Train_Label_dom.append(emo_dom[i])
            Train_Label_val.append(emo_val[i])
    Train_Data = np.array(Train_Data)
    Train_Label_act = np.array(Train_Label_act)
    Train_Label_dom = np.array(Train_Label_dom)
    Train_Label_val = np.array(Train_Label_val)
    
    # Feature Normalization Parameters
    eps = 1e-10 # assign small eps to prevent std=0 case
    Feat_mean_All = np.mean(Train_Data,axis=0)
    Feat_std_All = np.std(Train_Data,axis=0)
    Feat_std_All[np.where(Feat_std_All==0)[0]] = eps
    savemat('./NormTerm_Text/feat_norm_means.mat', {'normal_para':Feat_mean_All})
    savemat('./NormTerm_Text/feat_norm_stds.mat', {'normal_para':Feat_std_All})
    Label_mean_Act = np.mean(Train_Label_act)
    Label_std_Act = np.std(Train_Label_act)
    savemat('./NormTerm_Text/act_norm_means.mat', {'normal_para':Label_mean_Act})
    savemat('./NormTerm_Text/act_norm_stds.mat', {'normal_para':Label_std_Act})    
    Label_mean_Dom = np.mean(Train_Label_dom)
    Label_std_Dom = np.std(Train_Label_dom)    
    savemat('./NormTerm_Text/dom_norm_means.mat', {'normal_para':Label_mean_Dom})
    savemat('./NormTerm_Text/dom_norm_stds.mat', {'normal_para':Label_std_Dom})    
    Label_mean_Val = np.mean(Train_Label_val)
    Label_std_Val = np.std(Train_Label_val)      
    savemat('./NormTerm_Text/val_norm_means.mat', {'normal_para':Label_mean_Val})
    savemat('./NormTerm_Text/val_norm_stds.mat', {'normal_para':Label_std_Val})    
    
    