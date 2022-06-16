#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:55:20 2022

@author: crodrig1
"""



from transformers import TFRobertaForTokenClassification, RobertaTokenizer, TFRobertaModel
import shutil
MODEL_NAME = 'projecte-aina/roberta-base-ca-cased-ner'

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained('./{}_tokenizer/'.format(MODEL_NAME))

# just in case if there is no TF/Keras file provided in the model
# we can just use `from_pt` and convert PyTorch to TensorFlow
try:
  print('try downloading TF weights')
  model = TFRobertaForTokenClassification.from_pretrained(MODEL_NAME)
except:
  print('try downloading PyTorch weights')
  model = TFRobertaForTokenClassification.from_pretrained(MODEL_NAME, from_pt=True)

model.save_pretrained("./{}".format(MODEL_NAME), saved_model=True)





asset_path = '{}/saved_model/1/assets'.format(MODEL_NAME)

# let's save the vocab as txt file
with open('{}_tokenizer/vocab.txt'.format(MODEL_NAME), 'w') as f:
    for item in tokenizer.get_vocab().keys():
        f.write("%s\n" % item)

# let's copy both vocab.txt and merges.txt files to saved_model/1/assets
shutil.copy(MODEL_NAME+'_tokenizer/vocab.txt',asset_path)

shutil.copy(MODEL_NAME+'_tokenizer/merges.txt',asset_path)

#asset_path = '{}/saved_model/1/assets'.format(MODEL_NAME)




# get label2id dictionary 
labels = model.config.label2id
# sort the dictionary based on the id
labels = sorted(labels, key=labels.get)

with open(asset_path+'/labels.txt', 'w') as f:
    f.write('\n'.join(labels))

TFmodelfrompath = TFRobertaForTokenClassification.from_pretrained(MODEL_NAME)


## For Use in Spark



import sparknlp
# let's start Spark with Spark NLP
spark = sparknlp.start()

from sparknlp.annotator import *

tokenClassifier = RoBertaForTokenClassification\
  .loadSavedModel('{}/saved_model/1'.format(MODEL_NAME), spark)\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")\
  .setCaseSensitive(True)\
  .setMaxSentenceLength(128)

tokenClassifier.write().overwrite().save("./{}_spark_nlp".format(MODEL_NAME))


# Convert RoberTA general model

MODEL_NAME = 'projecte-aina/roberta-base-ca-v2'

# let's keep the tokenizer variable, we need it later
#tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
tokenizer = RobertaTokenizer.from_pretrained('projecte-aina/roberta-base-ca-cased-ner')
# let's save the tokenizer
tokenizer.save_pretrained('./{}_tokenizer/'.format(MODEL_NAME))

# just in case if there is no TF/Keras file provided in the model
# we can just use `from_pt` and convert PyTorch to TensorFlow
MODEL = "/home/crodrig1/sparknlp/roberta-base-ca-v2/"
try:
  print('try downloading TF weights')
  model = TFRobertaModel.from_pretrained(MODEL_NAME)
except:
  print('try downloading PyTorch weights')
  model = TFRobertaModel.from_pretrained(MODEL, from_pt=True)

model.save_pretrained("./{}".format(MODEL_NAME), saved_model=True)

asset_path = '{}/saved_model/1/assets'.format(MODEL_NAME)
#asset_path = '/home/crodrig1/sparknlp/sparknlp_ca/PlanTL-GOB-ES/roberta-base-ca/saved_model/1/assets'.format('roberta-base')


# with open('/home/crodrig1/sparknlp/sparknlp_ca/PlanTL-GOB-ES/roberta-base-ca_tokenizer/vocab.txt'.format('roberta-base'), 'w') as f:
#     for item in tokenizer.get_vocab().keys():
#         f.write("%s\n" % item)
with open('{}_tokenizer/vocab.txt'.format(MODEL_NAME), 'w') as f:
    for item in tokenizer.get_vocab().keys():
        f.write("%s\n" % item)
        
shutil.copy(MODEL_NAME+'_tokenizer/vocab.txt',asset_path)

shutil.copy(MODEL_NAME+'_tokenizer/merges.txt',asset_path)

#asset_path = '{}/saved_model/1/assets'.format(MODEL_NAME)




roberta_ca = RoBertaEmbeddings.loadSavedModel(
     '{}/saved_model/1'.format('roberta-base'),
     spark
 )\
 .setInputCols(["sentence",'token'])\
 .setOutputCol("embeddings")\
 .setCaseSensitive(True)\
 .setDimension(512)\
 .setStorageRef('roberta_base_ca') 
    
roberta_ca.write().overwrite().save("./{}_spark_nlp".format('roberta-base'))