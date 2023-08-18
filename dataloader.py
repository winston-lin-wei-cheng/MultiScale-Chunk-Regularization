#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wei-Cheng (Winston) Lin

Dataloader script for handling data I/O.
"""
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils import prepare_AlignEmoSet, LexicalChunkSplitData_rdnVFVC


class MspPodcastDataset_SpeechText(Dataset):
    """MSP-Podcast Emotion dataset"""

    def __init__(self, root_dir, label_dir, split_set):
        # parameters
        self.root_speech, self.root_text = root_dir
        
        # loading data paths and labels
        self._paths, self._labels_act, self._labels_dom, self._labels_val, self._aligns, _ = prepare_AlignEmoSet(label_dir, split_set=split_set)
        
        # setup norm folders
        norm_parameters_speech = 'NormTerm_Speech' # model: wav2vec2-large-robust (1024D)
        norm_parameters_text = 'NormTerm_Text'     # model: RoBERTa (768D)
            
        # loading norm-feature
        self.Feat_mean_Speech = loadmat('./'+norm_parameters_speech+'/feat_norm_means.mat')['normal_para']
        self.Feat_std_Speech = loadmat('./'+norm_parameters_speech+'/feat_norm_stds.mat')['normal_para']
        self.Feat_mean_Text = loadmat('./'+norm_parameters_text+'/feat_norm_means.mat')['normal_para']
        self.Feat_std_Text = loadmat('./'+norm_parameters_text+'/feat_norm_stds.mat')['normal_para']
        
        # loading norm-label
        self.Label_mean_act = loadmat('./'+norm_parameters_speech+'/act_norm_means.mat')['normal_para'][0][0]
        self.Label_std_act = loadmat('./'+norm_parameters_speech+'/act_norm_stds.mat')['normal_para'][0][0]
        self.Label_mean_dom = loadmat('./'+norm_parameters_speech+'/dom_norm_means.mat')['normal_para'][0][0]
        self.Label_std_dom = loadmat('./'+norm_parameters_speech+'/dom_norm_stds.mat')['normal_para'][0][0]
        self.Label_mean_val = loadmat('./'+norm_parameters_speech+'/val_norm_means.mat')['normal_para'][0][0]
        self.Label_std_val = loadmat('./'+norm_parameters_speech+'/val_norm_stds.mat')['normal_para'][0][0]

    def __len__(self):
        return len(self._paths)
    
    def __getitem__(self, idx):
        # loading audio-text data
        data_speech = loadmat(self.root_speech + self._paths[idx].replace('.wav','.mat'))['Audio_data']
        data_text = loadmat(self.root_text + self._paths[idx].replace('.wav','.mat'))['Text_data']
        
        # generate time-stamp for audio data
        data_speech_time = np.arange(0, len(data_speech)*0.02, 0.02) # wav2vec2 uses hop-size=20ms
        
        # clip time-seq to make sure it strictly matches the data size
        if len(data_speech_time)!=len(data_speech):
            data_speech_time = data_speech_time[:len(data_speech)]

        # z-norm and bounded in -3~3 range (i.e., 99.5% values coverage)
        data_speech = (data_speech-self.Feat_mean_Speech)/self.Feat_std_Speech
        data_speech[np.isnan(data_speech)]=0
        data_speech[data_speech>3]=3
        data_speech[data_speech<-3]=-3
        
        data_text = (data_text-self.Feat_mean_Text)/self.Feat_std_Text
        data_text[np.isnan(data_text)]=0
        data_text[data_text>3]=3
        data_text[data_text<-3]=-3
        
        # compute the word-chunk for audio data
        word_alignments = self._aligns[idx]
        word_data_chunk_speech = LexicalChunkSplitData_rdnVFVC(data_speech, data_speech_time, word_alignments, m=50) # m= 50frames * 20ms= 1sec chunk size 
        word_data_chunk_text = data_text
        
        # Label Normalization
        label_act = (self._labels_act[idx]-self.Label_mean_act)/self.Label_std_act
        label_dom = (self._labels_dom[idx]-self.Label_mean_dom)/self.Label_std_dom
        label_val = (self._labels_val[idx]-self.Label_mean_val)/self.Label_std_val

        return word_data_chunk_speech, word_data_chunk_text, label_act, label_dom, label_val
