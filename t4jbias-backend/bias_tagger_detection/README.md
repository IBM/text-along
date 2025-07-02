# Tagger model based on wikipidia edits data

Directory contains code from the paper, "[Automatically Neutralizing Subjective Bias in Text](https://arxiv.org/abs/1911.09709)".

Concretely this means algorithms for
* Identifying subjective biased words in sentences.

## Training Quickstart

### Option 1
These commands will download data and train the tagger model before running inference.
```
$ sh download_data_and_run_training.sh
```
OR 

To run on the CCC, use the following example to submit an interactive job:
```
jbsub -name 'tagger-train' -q x86_12h -cores 4+1 -require v100 -mem 32g -out joboutput.txt sh /u/$USER/media-bias-t4j-4374/src/bias_tagger_detection/download_data_and_run_training.sh
```
### Option 2

Directly by using the following code. This code assumes you already have the bias_data downloaded and in current path:
```
cd <project>/src/bias_tagger_detection
```
then 
```
python tagging/train.py --train ./bias_data/WNC/biased.word.train --test ./bias_data/WNC/biased.word.test --categories_file ./bias_data/WNC/revision_topics.csv --extra_features_top --pre_enrich --activation_hidden --category_input --learning_rate 0.0003 --epochs 10 --debias_weight 1.3 --working_dir detection_outputs/ --bert_word_embeddings
```
## Testing the detection model 

You can use the following command, please replace `<path>` with the path of the directory where the model is:
```
python tagging/taggertest.py --tagger_checkpoint <path>/tagger_model_2.ckpt --working_dir TEST --categories_file ./bias_data/WNC/revision_topics.csv --extra_features_top --pre_enrich --activation_hidden --category_input
```

## Data

Click [this link to download](http://bit.ly/bias-corpus) (100MB, expands to 500MB). 

If that link is broken, try this mirror: [download link](https://www.dropbox.com/s/qol3rmn0rq0dfhn/bias_data.zip?dl=0)

## Overview

`harvest/`: Code for making the dataset. It works by crawling and filtering Wikipedia for bias-driven edits.

`tagging/`: Code for training models and using trained models to run inference.


### **acknowledgements.txt**

This file contains acknowledgements to outside data sources as requested by the authors (bias-lexicon.txt for example).