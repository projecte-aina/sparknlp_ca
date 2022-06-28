#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 08:04:57 2022

@author: crodrig1
"""

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.base import LightPipeline

from pyspark.ml import Pipeline
spark = sparknlp.start()



text = "A partir del any vint, l'incidència delicada als virus s'hauria d'abaixar, fent-la passar del 21,1 per cent l'any 1999 al 19,4 per cent de 2000. Parlem-ne."
data = spark.createDataFrame([[text]]).toDF("text")

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
  .pretrained("sentence_detector_dl", "xx") \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")



tokenizer = Tokenizer() \
    .setInputCols(['sentence']).setOutputCol('token')\
    .setInfixPatterns(["(d|D)(els)","(d|D)(el)","(a|A)(ls)","(a|A)(l)","(p|P)(els)","(p|P)(el)","([A-zÀ-ú])(-[A-zÀ-ú]+)","(d'|D')([A-zÀ-ú]+)","(l'|L')([A-zÀ-ú]+)", "(l'|l'|s'|s'|d'|d'|m'|m'|D'|D'|L'|L'|S'|S'|N'|N'|M'|M')([A-zÀ-ú]+)", "([A-zÀ-ú]+)(\.|,)","([A-zÀ-ú]+)(ls|'l|'ns|'t|'m|'n|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)(\.|,|;)+"])\
    

    


pipeline = Pipeline().setStages([documentAssembler, sentencerDL,  tokenizer]).fit(data)
light_model = LightPipeline(pipeline)


light_result = light_model.annotate(text)

import pandas as pd

result = pd.DataFrame(zip(light_result['token']), columns = ["token"])

print(result)


