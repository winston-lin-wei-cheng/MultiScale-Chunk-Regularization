#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 17:34:22 2021

@author: winston
"""
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import AutoModelForPreTraining
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from scipy.io import savemat, loadmat
from sklearn.decomposition import PCA


def wav2vec_feat_extract768(fname):
    signal, rate  = librosa.load(fname, sr=16000)
    input_data = processor(signal, return_tensors="pt").input_values  # Batch size 1
    hidden_feats = model(input_data).last_hidden_state
    hidden_feats = hidden_feats.squeeze(0)
    hidden_feats = hidden_feats.data.cpu().numpy()
    return hidden_feats

def wav2vec_feat_extract256(fname):
    signal, rate  = librosa.load(fname, sr=16000)
    input_data = processor(signal, return_tensors="pt").input_values  # Batch size 1
    hidden_feats = model(input_data)['projected_states']
    hidden_feats = hidden_feats.squeeze(0)
    hidden_feats = hidden_feats.data.cpu().numpy()
    return hidden_feats

def wav2vec_feat_extract1024(fname):
    signal, rate  = librosa.load(fname, sr=16000)
    input_data = processor(signal, return_tensors="pt").input_values  # Batch size 1
    hidden_feats = model(input_data.to("cuda")).last_hidden_state
    hidden_feats = hidden_feats.squeeze(0)
    hidden_feats = hidden_feats.data.cpu().numpy()
    return hidden_feats
###############################################################################



# # load the pretrained model (for 768-dim wav2vec2)
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


# # load the pretrained model (for 256-dim wav2vec2, having additional FCN projection layer)
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
# model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base")


# load the pretrained model (for 1024-dim wav2vec2-large)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-robust").to("cuda")


# dataset dir
input_path = './MSP-IMPROV/Audio/'
output_path = './MSP-IMPROV/Features/Wav2Vec1024/feat_mat/'

# creating saving repo
if not os.path.isdir(output_path):
    os.makedirs(output_path)

# feature extraction process
ERROR_record = ''
for root, directories, files in os.walk(input_path):
    files = sorted(files)
    for filename in files:
        # Join the two strings in order to form the full filepath.
        filepath = os.path.join(root, filename)
        if '.wav' in filepath:
            try:
                # features = wav2vec_feat_extract768(filepath)
                # features = wav2vec_feat_extract256(filepath)
                features = wav2vec_feat_extract1024(filepath)
                filename = filename.replace('wav','mat')
                savemat(os.path.join(output_path, filename), {'Audio_data':features})
            except:
                ERROR_record += 'Error: '+filename+'\n'
            
        else:
            raise ValueError("Unsupport File Type!!")

record_file = open("ErrorRecord_Wav2Vec.txt","w") 
record_file.write(ERROR_record)
record_file.close()


###############################################################################
###############################################################################


# # construct PCA model for mapping wav2vec2 features into dense space
# input_path = './MSP-PODCAST-Publish-1.10/Features/Wav2Vec768/feat_mat/'
# output_path = './MSP-PODCAST-Publish-1.10/Features/Wav2Vec768_PCA128/feat_mat/'

# # All_Data = []
# # for root, directories, files in os.walk(input_path):
# #     files = sorted(files)
# #     for filename in files:
# #         # Join the two strings in order to form the full filepath.
# #         filepath = os.path.join(root, filename)
# #         data = loadmat(filepath)['Audio_data']
# #         data = np.mean(data,axis=0) # use mean-vector as the sentence-representation for PCA
# #         All_Data.append(data)
# # All_Data = np.array(All_Data)
# # pca = PCA(n_components=128, random_state=719)
# # pca.fit(All_Data)
# # # save the trained PCA model
# # pickle.dump(pca, open('./MSP-PODCAST-Publish-1.10/Features/Wav2Vec768_PCA128/PCA_model.pkl','wb'))

# # load the trained PCA model
# pca = pickle.load(open('./MSP-PODCAST-Publish-1.10/Features/Wav2Vec768_PCA128/PCA_model.pkl','rb'))
# for root, directories, files in os.walk(input_path):
#     files = sorted(files)
#     for filename in files:
#         filepath = os.path.join(root, filename)
#         features = loadmat(filepath)['Audio_data']
#         pca_features = pca.transform(features) # dense space: to 128D
#         savemat(os.path.join(output_path, filename), {'Audio_data':pca_features})

