# Multi-Scale Chunk Regularization for Audio-Text Emotion Recognition
This is a Pytorch implementation of the paper: [Enhancing Resilience to Missing Data in Audio-Text Emotion Recognition with Multi-Scale Chunk Regularization](https://ieeexplore.ieee.org/XXX). The experiments and trained models were based on the MSP-Podcast v1.10 corpus & pretrained *wav2vec2-large-robust* (audio) and *RoBERTa-base* (text) features in the paper.

![The procedure of retrieving chunk-level local emotions by emo-rankers](/images/XXX.png)


# Suggested Environment and Requirements
1. Python 3.6+
2. Ubuntu 18.04
3. CUDA 10.0+
4. pytorch version 1.4.0+
5. huggingface transformer version 4.10.0+
6. The scipy, numpy and pandas...etc standard packages
7. The MSP-Podcast v1.10 corpus (request to download from [UTD-MSP lab website](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html))


# Data Preparation & Feature Extraction
After extracting the 130-dim LLDs (via OpenSmile) for the corpus. Then, use the **norm_para.py** to save normalization parameters for z-norm of the input data and label. We have provided the parameters of v1.6 corpus in the *'NormTerm'* folder.


# Emotion Rankers
Codes for building the chunk-level emotion rankers are put in the *'chunk_emotion_rankers'* folder. The **generate_preference_labels.py** is used for generating preference labels (based on the QA approach) for training. We have provided the trained rankers in the *'trained_ranker_model_v1.6'* folder. And the retrieved emotion ranking sequences for the v1.6 corpus are in the *'EmoChunkRankSeq'* folder. If users like to train the rankers and retrieve sequences from scratch, you can follow the steps:
1. use **generate_preference_labels.py** to obtain preference labels
2. run **training_ranker.py** in the terminal to train the model
```
python training_ranker.py -ep 20 -batch 128 -emo Dom
```
3. run **testing_ranker.py** in the terminal to test the model (optional)
```
python testing_ranker.py -ep 20 -batch 128 -emo Dom
```
4. run **generate_ranking_seqence.py** in the terminal to retrieve ranking sequences
```
python generate_ranking_seqence.py -ep 20 -batch 128 -emo Dom -set Train
```


# Seq2Seq SER Models
After retrieved the chunk-level ranking sequences, it's straightforward to directly treat them as target sequence to train the Seq2Seq SER model. We use the *'generate_chunk_EmoSeq'* function in **utils.py** to track emo-trends, smooth and re-scale to generate the target emotion curves for training. Simply run the following steps to build the model: 
1. run **training.py** in the terminal to train the model
```
python training.py -ep 30 -batch 128 -emo Val
```
2. run **testing.py** in the terminal to test the model (optional)
```
python testing.py -ep 30 -batch 128 -emo Val
```

We also provide the trained models in the *'trained_seq2seq_model_v1.6'*.The CCC performances of models based on the test set are shown in the following table. Note that the results are slightly different from the [paper](https://ieeexplore.ieee.org/XXX) since we performed statistical test in the paper (i.e., we averaged multiple trails results together).

|  | Aro. | Dom. | Val. |
|:----------------:|:----------------:|:----------------:|:----------------:|
| Seq2Seq-RankerInfo | 0.7103 | 0.6302 | 0.3222 |


Users can get these results by running the **testing.py** with corresponding args.


# Reference
If you use this code, please cite the following paper:

Wei-Cheng Lin and Carlos Busso, "Sequential Modeling by Leveraging Non-Uniform Distribution of Speech Emotion"

```
@article{Lin_202x_2,
 	author = {W.-C. Lin and C. Busso},
	title = {Sequential Modeling by Leveraging Non-Uniform Distribution of Speech Emotion},
	journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
	volume = {To appear},
	number = {},
	year = {2023},
	pages = {},
 	month = {},
 	doi={},
}
```
