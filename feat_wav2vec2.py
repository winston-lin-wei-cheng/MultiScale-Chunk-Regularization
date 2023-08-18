#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wei-Cheng (Winston) Lin

This script extracts the pretrained wav2vec2-large feature embeddings w.r.t the input audio files.
"""
import os
import librosa
from scipy.io import savemat
from transformers import Wav2Vec2Processor, Wav2Vec2Model


def wav2vec_feat_extract1024(fname):
    signal, rate  = librosa.load(fname, sr=16000)
    input_data = processor(signal, return_tensors="pt").input_values  # Batch size 1
    hidden_feats = model(input_data.to("cuda")).last_hidden_state
    hidden_feats = hidden_feats.squeeze(0)
    hidden_feats = hidden_feats.data.cpu().numpy()
    return hidden_feats
###############################################################################



# I/O paths
input_path = './MSP-PODCAST-Publish-1.10/Audio/'
output_path = './MSP-PODCAST-Publish-1.10/Features/Wav2Vec1024/feat_mat/'

# creating saving repo
if not os.path.isdir(output_path):
    os.makedirs(output_path)

# load the pretrained model (for 1024-dim wav2vec2-large)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-robust").to("cuda")

# feature extraction process
ERROR_record = ''
for root, directories, files in os.walk(input_path):
    files = sorted(files)
    for filename in files:
        # Join the two strings in order to form the full filepath.
        filepath = os.path.join(root, filename)
        if '.wav' in filepath:
            try:
                features = wav2vec_feat_extract1024(filepath)
                filename = filename.replace('wav','mat')
                savemat(os.path.join(output_path, filename), {'Audio_data':features})
            except:
                ERROR_record += 'Error: '+filename+'\n'
            
        else:
            raise ValueError("Unsupport File Type!!")

record_file = open("ErrorRecord_Wav2Vec2.txt","w") 
record_file.write(ERROR_record)
record_file.close()
