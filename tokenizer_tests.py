#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 08:04:57 2022

@author: crodrig1
"""
##https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.Tokenizer.html
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *


from pyspark.ml import Pipeline
spark = sparknlp.start()
data = spark.createDataFrame([["A partir d'aquest any, la incidència s'hauria de baixar, fent-la passar del 21,1 per cent l'any 1999 al 19,4 per cent de 2000. Parlem-ne."]]).toDF("text")

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
  .pretrained("sentence_detector_dl", "xx") \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")

ex_list = ["aprox.","pàg.","p.ex.","gen.","feb.","abr.","jul.","set.","oct.","nov.","dec.","dr.","dra.","sr.","sra.","srta.","núm.","st.","sta.","pl.","etc.", "ex."]#,"’", '”', "(", "[", "l'","l’","s'","s’","d’","d'","m’","m'","L'","L’","S’","S'","N’","N'","M’","M'"]
ex_list_all = []
ex_list_all.extend(ex_list)
ex_list_all.extend([x[0].upper() + x[1:] for x in ex_list])
ex_list_all.extend([x.upper() for x in ex_list])



# tokenització correcte:
# [A, partir, d', aquest, any, ,, la, incidència, s', hauria, de, baixar, ,, fent, -la, passar, del, 21,1, per, cent, l', any, 1999, al, 19,4, per, cent, de, 2000, ., Parlem, -ne, .]

tokenizer = Tokenizer() \
    .setInputCols(['sentence']).setOutputCol('token') \
    .setPrefixPattern("\A(l'|l’|s'|s’|d’|d'|m’|m'|L'|L’|S’|S'|N’|N'|M’|M')") \

#'.|,|;|:|!|?|*|-|(|)|"|\''
pipeline = Pipeline().setStages([documentAssembler, sentencerDL,  tokenizer]).fit(data)

result = pipeline.transform(data)

result.selectExpr("token.result").show(truncate=False)


