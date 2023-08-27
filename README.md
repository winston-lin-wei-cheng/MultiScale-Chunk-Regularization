# Multi-Scale Chunk Regularization for Audio-Text Emotion Recognition
This is a Pytorch implementation of the paper: [Enhancing Resilience to Missing Data in Audio-Text Emotion Recognition with Multi-Scale Chunk Regularization](https://ieeexplore.ieee.org/XXX). The experiments and trained models were based on the MSP-Podcast v1.10 corpus & pretrained *wav2vec2-large-robust* (audio) and *RoBERTa-base* (text) features in the paper.

![The full framework of the porposed model](/images/framework.png)


# Suggested Environment and Requirements
1. Python 3.6+
2. Ubuntu 18.04+
3. pytorch version 1.9.0+cu102
4. huggingface transformer version 4.5.1
5. [textgrid](https://pypi.org/project/TextGrid/)
6. The scipy, numpy and pandas...etc standard packages
7. The MSP-Podcast v1.10 corpus (request to download from [UTD-MSP lab website](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html))


# How to Run
1. Place the downloaded MSP-Podcast v1.10 corpus under the same directory.
2. Extract the pretrained deep features using **feat_wav2vec2.py** (for audio) and **feat_roberta.py** (for text), the outputs will be saved in the created *'Features'* folder.
3. [Optional] Run **norm_para.py** to obtain parameters for z-norm, the outputs will be saved in the created *'NormTerm_Speech'* and *'NormTerm_Text'* folders.
4. Run the main script,
```
python main.py
```
The testing results (in terms of CCC) of the trained model will be directly printed out after the training is done.


# Reference
If you use this code, please cite the following paper:

Wei-Cheng Lin, Lucas Goncalves and Carlos Busso, "Enhancing Resilience to Missing Data in Audio-Text Emotion Recognition with Multi-Scale Chunk Regularization"

```
@InProceedings{Lin_2023_3, 
	author={W.-C. Lin and L. Goncalves and C. Busso}, 
	title={Enhancing Resilience to Missing Data in Audio-Text Emotion Recognition with Multi-Scale Chunk Regularization},
	booktitle={ACM International Conference on Multimodal Interaction (ICMI 2023)},  
	volume={To appear},
	year={2023}, 
	month={October}, 
	address =  {Paris, France},
	pages={}, 
	doi={},
}
```
