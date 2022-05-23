#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:55:20 2022

@author: crodrig1
"""



from transformers import TFRobertaForSequenceClassification, RobertaTokenizer 
import shutil
MODEL_NAME = 'projecte-aina/roberta-base-ca-cased-tc'

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained('./{}_tokenizer/'.format(MODEL_NAME))

# just in case if there is no TF/Keras file provided in the model
# we can just use `from_pt` and convert PyTorch to TensorFlow
try:
  print('try downloading TF weights')
  model = TFRobertaForSequenceClassification.from_pretrained(MODEL_NAME)
except:
  print('try downloading PyTorch weights')
  model = TFRobertaForSequenceClassification.from_pretrained(MODEL_NAME, from_pt=True)

model.save_pretrained("./{}".format(MODEL_NAME), saved_model=True)





asset_path = '{}/saved_model/1/assets'.format(MODEL_NAME)

# let's save the vocab as txt file
with open('{}_tokenizer/vocab.txt'.format(MODEL_NAME), 'w') as f:
    for item in tokenizer.get_vocab().keys():
        f.write("%s\n" % item)

# let's copy both vocab.txt and merges.txt files to saved_model/1/assets
shutil.copy(MODEL_NAME+'_tokenizer/vocab.txt',asset_path)

shutil.copy(MODEL_NAME+'_tokenizer/merges.txt',asset_path)

asset_path = '{}/saved_model/1/assets'.format(MODEL_NAME)




# get label2id dictionary 
labels = model.config.label2id
# sort the dictionary based on the id
labels = sorted(labels, key=labels.get)

with open(asset_path+'/labels.txt', 'w') as f:
    f.write('\n'.join(labels))

TFmodelfrompath = TFRobertaForSequenceClassification.from_pretrained(MODEL_NAME)
