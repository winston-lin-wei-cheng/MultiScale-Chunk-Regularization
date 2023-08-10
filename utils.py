#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:44:00 2019

@author: winston
"""
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import torch
import textgrid
import random
np.random.seed(794)
# random.seed(12)


# let's try MTL framework this time!!
def getPaths(path_label, split_set):
    """
    This function is for filtering data by different constraints of label
    Args:
        path_label$ (str): path of label.
        split_set$ (str): 'Train', 'Development' or 'Test' are supported.
    """
    label_table = pd.read_csv(path_label)
    whole_fnames = (label_table['FileName'].values).astype('str')
    split_sets = (label_table['Split_Set'].values).astype('str')
    emo_act = label_table['EmoAct'].values
    emo_dom = label_table['EmoDom'].values
    emo_val = label_table['EmoVal'].values
    _paths = []
    _label_act = []
    _label_dom = []
    _label_val = []
    for i in range(len(whole_fnames)):
        # Constrain with Split Sets      
        if split_sets[i]==split_set:
            # Constrain with Emotional Labels
            _paths.append(whole_fnames[i])
            _label_act.append(emo_act[i])
            _label_dom.append(emo_dom[i])
            _label_val.append(emo_val[i])
        else:
            pass
    return np.array(_paths), np.array(_label_act), np.array(_label_dom), np.array(_label_val)

def evaluation_metrics(true_value,predicted_value):
    corr_coeff = np.corrcoef(true_value,predicted_value)
    ccc = 2*predicted_value.std()*true_value.std()*corr_coeff[0,1]/(predicted_value.var() + true_value.var() + (predicted_value.mean() - true_value.mean())**2)
    return(ccc,corr_coeff)

def cc_coef(output, target):
    mu_y_true = torch.mean(target)
    mu_y_pred = torch.mean(output)                                                                                                                                                                                              
    return 1 - 2 * torch.mean((target - mu_y_true) * (output - mu_y_pred)) / (torch.var(target) + torch.var(output) + torch.mean((mu_y_pred - mu_y_true)**2))    

def prepare_AlignEmoSet(path_label, split_set):
    # PATH settings
    align_root_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.10/ForceAligned/'
    set_paths, set_labels_act, set_labels_dom, set_labels_val = getPaths(path_label, split_set)
    # Main function
    File_Name = []
    EmoAct_Rsl = []
    EmoDom_Rsl = []
    EmoVal_Rsl = []
    Align_Rsl = []
    for i in range(len(set_paths)):
        # load the word-alignment file 
        word_aligns = []
        align_rsl = textgrid.TextGrid.fromFile(align_root_dir+set_paths[i].replace('.wav','.TextGrid'))
        for j in range(len(align_rsl[0])): # word-level
            if align_rsl[0][j].mark!="":   # remove <eps> sound (i.e., silence)
                align_t_start = align_rsl[0][j].minTime
                align_t_end = align_rsl[0][j].maxTime
                duration = align_rsl[0][j].maxTime - align_rsl[0][j].minTime
                align_word = align_rsl[0][j].mark
                word_aligns.append((float(align_t_start), float(align_t_end), float(duration), align_word))
        # special treatment for those zero-words segment
        if len(word_aligns)==0:
            align_t_start = align_rsl[0][0].minTime
            align_t_end = align_rsl[0][0].maxTime
            duration = align_rsl[0][0].maxTime - align_rsl[0][0].minTime
            align_word = 'None'
            word_aligns.append((float(align_t_start), float(align_t_end), float(duration), align_word))
        # output parsing result
        File_Name.append(str(set_paths[i]))
        EmoAct_Rsl.append(set_labels_act[i])
        EmoDom_Rsl.append(set_labels_dom[i])
        EmoVal_Rsl.append(set_labels_val[i])
        Align_Rsl.append(word_aligns)
    # obtain the max words in a sentence across the entire set
    max_word_len = 0
    for i in range(len(Align_Rsl)):
        if len(Align_Rsl[i])>max_word_len:
            max_word_len = len(Align_Rsl[i])
    return np.array(File_Name), np.array(EmoAct_Rsl), np.array(EmoDom_Rsl), np.array(EmoVal_Rsl), Align_Rsl, max_word_len

# split original batch data into batch small-chunks data with
# proposed dynamic window step size which depends on the sentence duration 
def DynamicChunkSplitData(Online_data, m, C, n):
    """
    Note! This function can't process sequence length which less than given m
    (e.g., 1sec=62frames, if LLDs/Mel-Spec extracted by hop size 16ms then 16ms*62=0.992sec~=1sec)
    (e.g., 1sec=50frames, if wav2vec2 extracted by hop size 20ms then 20ms*50=1sec)
    Please make sure all your input data's length are greater then given m.
    
    Args:
         Online_data$ (list): list of data array for a single sentence
                   m$ (int) : chunk window length (i.e., number of frames within a chunk)
                   C$ (int) : number of chunks splitted for a sentence
                   n$ (int) : scaling factor to increase number of chunks splitted in a sentence
    """
    num_shifts = n*C-1  # Tmax = 11sec (for the MSP-Podcast corpus), 
                        # chunk needs to shift 10 times to obtain total C=11 chunks for each sentence
    Split_Data = []
    for i in range(len(Online_data)):
        data = Online_data[i]
        # window-shifting size varied by differenct length of input utterance => dynamic step size
        step_size = int(int(len(data)-m)/num_shifts)      
        # Calculate index of chunks
        start_idx = [0]
        end_idx = [m]
        for iii in range(num_shifts):
            start_idx.extend([start_idx[0] + (iii+1)*step_size])
            end_idx.extend([end_idx[0] + (iii+1)*step_size])    
        # Output Split Data
        for iii in range(len(start_idx)):
            Split_Data.append( data[start_idx[iii]: end_idx[iii]] )    
    return np.array(Split_Data)

def RdnMultiRslChunk(data, sample_pool=[0.2, 0.4, 0.6, 0.8, 1.0]):
    """
    This function random crops of the input data chunks to obtain multi-resolution chunk sizes for training. 
    After cropping the data, it would pad with zeros to maintain the same data shape.
    Args:
            data$ (np.arry): the output of "DynamicChunkSplitData" function, shape=[C, m, feat-dim]
        sample_pool$ (list): list of sampling chunk-size options in secs
    """
    _pool = sample_pool
    MultiRslDataChunk = []
    Actual_Length = []
    for i in range(len(data)):
        sample_size = random.choice(_pool)
        if sample_size==1: # keeping the full 1sec data chunk => no crops performed!
            MultiRslDataChunk.append(data[i])
            Actual_Length.append(len(data[i]))
        else:
            frames = int(sample_size*len(data[i]))
            rdn_center_frame = random.randrange(0+int(frames/2), len(data[i])-int(frames/2))
            rdn_crop_data = data[i][rdn_center_frame-int(frames/2):rdn_center_frame+int(frames/2)]
            Actual_Length.append(len(rdn_crop_data))
            paddings = np.zeros((len(data[i])-len(rdn_crop_data),len(data[i].T)))
            MultiRslDataChunk.append(np.concatenate((rdn_crop_data,paddings),axis=0))
    return np.array(MultiRslDataChunk), Actual_Length

# split original sentence data into small data chunks based on the number of words (varied C),
# then, each data chunk is randomly cropped (varied F) in "main.py"- collate_fn
def LexicalChunkSplitData_rdnVFVC(data, data_time, word_alignments, m):
    if len(word_alignments)<2:
        # C at least >=2, some sentences only have single/no words
        return DynamicChunkSplitData([data], m=m, C=2, n=1)
    else:
        return DynamicChunkSplitData([data], m=m, C=len(word_alignments), n=1)

# parymid temporal pooling operation to deal with varied sizes chunk-inputs
# we fixed the pooling levels=[1, 2, 4] for now
def ParymidTempPool(data, actual_lengths=None):
    """
    Args:
               data$ (np.arry): data matrix with shape= [Word-Chunks, Frames, Feats]
        actual_lengths$ (list): list of padding indexes for each word-chunk,
                   if given None, this function will assume the input data does not contain paddings.
    """
    Sent_Pool_Rsl = []
    for w_idx in range(len(data)): # word index for each sentence
        if not actual_lengths: # FF case
            actual_data = data[w_idx]
        else: # VF case, required actual length index for each word chunk
            actual_data = data[w_idx]
            actual_data = actual_data[:actual_lengths[w_idx]]
        if len(actual_data)>=5: # set min eps(=5) frames for the chunk data
            # concat word(chunk)-level temporal parymid pooling result
            pool_rsl = []
            # level-1 pooling
            pool_rsl.append(np.mean(actual_data,axis=0))
            # level-2 pooling
            _split_length = int(len(actual_data)/2)
            pool_rsl.append(np.mean(actual_data[:_split_length],axis=0))
            pool_rsl.append(np.mean(actual_data[_split_length:],axis=0))
            # level-4 pooling
            _split_length = int(len(actual_data)/4)
            pool_rsl.append(np.mean(actual_data[:1*_split_length],axis=0))
            pool_rsl.append(np.mean(actual_data[1*_split_length:2*_split_length],axis=0))
            pool_rsl.append(np.mean(actual_data[2*_split_length:3*_split_length],axis=0))
            pool_rsl.append(np.mean(actual_data[3*_split_length:],axis=0))
            Sent_Pool_Rsl.append(np.array(pool_rsl).reshape(-1))
        else:
            # concat word(chunk)-level temporal parymid pooling result
            pool_rsl = []
            # level-1 pooling* 7 times
            pool_rsl.append(np.mean(actual_data,axis=0))
            pool_rsl.append(np.mean(actual_data,axis=0))
            pool_rsl.append(np.mean(actual_data,axis=0))
            pool_rsl.append(np.mean(actual_data,axis=0))
            pool_rsl.append(np.mean(actual_data,axis=0))
            pool_rsl.append(np.mean(actual_data,axis=0))
            pool_rsl.append(np.mean(actual_data,axis=0))
            Sent_Pool_Rsl.append(np.array(pool_rsl).reshape(-1))
    return np.array(Sent_Pool_Rsl)

def random_drop_frames(data, ratio):
    drop_idx = np.random.choice(len(data), size=int(len(data)*ratio), replace=False)
    data[drop_idx] = 0
    return data

def packet_loss_frames(data, prob_N, prob_L):
    """
    This function uses two-states Markov Chain to simulate the effect of packet loss for speech frames.
    Args:
        data$ (np.arry): speech data matrix with shape= [Frames, Feats]
        prob_N$ (float): probability of no-loss state
        prob_L$ (float): probability of loss state
    """
    Pn, Pl = prob_N, prob_L
    state = ["N", "L"]
    TransMat = np.array([[Pn, 1-Pn], [1-Pl, Pl]])
    # init state is always at the "N" (no-loss)
    StartingState = 0
    CurrentState = StartingState
    PacketLoss_Mask = []
    for i in range(len(data)):
        # Deciding the next state using a random.choice()
        # function, that takes list of states and the probability
        # to go to the next states from our current state
        CurrentState = np.random.choice([0, 1], p=TransMat[CurrentState])
        # appending states to simulate packet losses
        PacketLoss_Mask.append(state[CurrentState])
    drop_idx = np.where(np.array(PacketLoss_Mask)=="L")[0]
    data[drop_idx] = 0
    return data

def random_drop_words(dataS, dataT, dataS_time, fname, aligns, ratio):
    token_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.10/RoBERTa_Tokens/'
    tokens = []
    with open(token_dir+fname) as f:
        _file_lines = f.readlines()
        for i in range(len(_file_lines)):
            tokens.append(_file_lines[i].strip())
    num_words_to_drop = int(np.round(ratio*len(aligns)))
    rdn_drop_idx = np.random.choice(len(aligns), size=num_words_to_drop, replace=False)
    # main: random drop words
    for iii in rdn_drop_idx:
        # drop the speech frames based on the exact word-level alignment info
        word_start, word_end = aligns[iii][0], aligns[iii][1]
        drop_sphF_start = np.where(abs(dataS_time-word_start)==min(abs(dataS_time-word_start)))[0][0]
        drop_sphF_end = np.where(abs(dataS_time-word_end)==min(abs(dataS_time-word_end)))[0][0]
        dataS[drop_sphF_start:drop_sphF_end+1] = 0
        # drop the text embd based on the most similar token
        drop_word = aligns[iii][-1]
        _highest_simscore = 0
        drop_token_idx = 0
        for jjj in range(len(tokens)):
            _score = SequenceMatcher(None, drop_word, tokens[jjj]).ratio()
            if _score > _highest_simscore:
                _highest_simscore = _score
                drop_token_idx = jjj
        dataT[drop_token_idx] = 0
    return dataS, dataT

