#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wei-Cheng (Winston) Lin

This script extracts the pretrained RoBERTa-base feature embeddings w.r.t the input text files.
    - English language
    - BERT-based arch.
    - The model is case-sensitive ("english" and "English" is different)
    - Note that this model is primarily aimed at being fine-tuned on tasks that use 
      the whole sentence to make decisions, such as sequence classification (i.e., for seq2one tasks).
    - The training data used for this model contains a lot of unfiltered content from the internet, 
      which is far from neutral. Therefore, the model can have biased predictions (gender, ethnicity...etc). 
"""
import os
import numpy as np
from scipy.io import savemat
from transformers import pipeline



# I/O paths
input_path = './MSP-PODCAST-Publish-1.10/Transcripts/'
output_path = './MSP-PODCAST-Publish-1.10/Features/RoBERTa768/feat_mat/'

# creating saving repo
if not os.path.isdir(output_path):
    os.makedirs(output_path)

# load the pretrained model (for 768-dim roberta-base) 
pipeline = pipeline('feature-extraction', model='roberta-base')

# feature extraction process
ERROR_record = ''
for root, directories, files in os.walk(input_path):
    files = sorted(files)
    for filename in files:
        # Join the two strings in order to form the full filepath.
        filepath = os.path.join(root, filename)
        if '.txt' in filepath:
            try:
                # load the text file 
                with open(filepath) as f:
                    text = f.readlines()
                    text = " ".join(text)
                    text = text.strip()
                # extract embeddings into (Time, feat_dim) 2D-array matrix format
                feature = pipeline(text)
                feature = np.array(feature)
                feature = np.squeeze(feature, axis=0)
                # save to the output folder
                filename = filename.replace('txt','mat')
                savemat(os.path.join(output_path, filename), {'Text_data':feature})
            except:
                ERROR_record += 'Error: '+filename+'\n'
            
        else:
            raise ValueError("Unsupport File Type!!")

record_file = open("ErrorRecord_RoBERTa.txt","w") 
record_file.write(ERROR_record)
record_file.close()
