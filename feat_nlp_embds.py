#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:51:19 2022

@author: winston
"""
import os
import numpy as np
from scipy.io import savemat
from transformers import pipeline, RobertaTokenizer


# # How to use this model to get the features of a given text in PyTorch
# from transformers import RobertaTokenizer, RobertaModel
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# model = RobertaModel.from_pretrained('roberta-base')
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)


"""
This script extracts the [pre-trained RoBERTa model] embeddings w.r.t the input text file.
    - English language
    - BERT-based arch.
    - The model is case-sensitive ("english" and "English" is different)
    - Note that this model is primarily aimed at being fine-tuned on tasks that use 
      the whole sentence to make decisions, such as sequence classification (i.e., for seq2one tasks).
    - The training data used for this model contains a lot of unfiltered content from the internet, 
      which is far from neutral. Therefore, the model can have biased predictions (gender, ethnicity...etc). 
"""

# PATH parameters
rootDir = './MSP-PODCAST-Publish-1.10/Transcripts/'
outDir = './MSP-PODCAST-Publish-1.10/Features/RoBERTa_Embedding768/feat_mat/'

# creating saving repo
if not os.path.isdir(outDir):
    os.makedirs(outDir)

# Loading the PreTrained Model for feature extraction 
pipeline = pipeline('feature-extraction', model='roberta-base')

for root, directories, files in os.walk(rootDir):
    files = sorted(files)
    for fname in files:
        # load the text file 
        with open(root+fname) as f:
            text = f.readlines()
            text = " ".join(text)
            text = text.strip()
        # extract embeddings into (Time, feat_dim) 2D-array matrix format
        feature = pipeline(text)
        feature = np.array(feature)
        feature = np.squeeze(feature, axis=0)
        # save to the output folder
        savemat(os.path.join(outDir, fname.replace('.txt','.mat')), {'Text_data':feature})

###############################################################################
###############################################################################
###############################################################################


# ###################################################################
# ### Get the corresponding tokens of the extracted text-features ###
# ###################################################################

# # PATH parameters
# rootDir = './MSP-PODCAST-Publish-1.10/Transcripts/'
# outDir = './MSP-PODCAST-Publish-1.10/RoBERTa_Tokens/'

# # creating saving repo
# if not os.path.isdir(outDir):
#     os.makedirs(outDir)

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# for root, directories, files in os.walk(rootDir):
#     files = sorted(files)
#     for fname in files:
#         # load the text file 
#         with open(root+fname) as f:
#             text = f.readlines()
#             text = " ".join(text)
#             text = text.strip()
#         # extract tokens
#         tkns = tokenizer.tokenize(text)
#         tkns.insert(0, "[CLS]")
#         tkns.append("[SEP]")
#         tkns_clean = []
#         for tkn in tkns:
#             if "Ġ" in tkn:
#                 tkns_clean.append(tkn.replace("Ġ", ""))
#             else:
#                 tkns_clean.append(tkn)
#         # save to the output folder
#         f = open(os.path.join(outDir, fname),'w')
#         for _tkn in tkns_clean:
#         	f.write(_tkn+"\n")
#         f.close()

