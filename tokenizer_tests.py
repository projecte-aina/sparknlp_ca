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
data = spark.createDataFrame([["el 26 de set. anem al c/ de l'arbre del sr. Minó i el Sr. Pepu. Anem-nos-en d'aquí, dona-n'hi tres."]]).toDF("text")

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

ex_list = ["aprox.","pàg.","p.ex.","gen.","feb.","abr.","jul.","set.","oct.","nov.","dec.","dr.","dra.","sr.","sra.","srta.","núm.","st.","sta.","pl.","etc.", "ex."]#,"’", '”', "(", "[", "l'","l’","s'","s’","d’","d'","m’","m'","L'","L’","S’","S'","N’","N'","M’","M'"]
ex_list_all = []
ex_list_all.extend(ex_list)
ex_list_all.extend([x[0].upper() + x[1:] for x in ex_list])
ex_list_all.extend([x.upper() for x in ex_list])

tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token") \
    .setSuffixPattern("([^\s\w]?)('hi|ls|'l|'ns|'t|'m|'n|-n|-en|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)\z")\
    .setExceptions(ex_list_all)#.fit(data)
tokenizer.addInfixPattern("([^\s\w]?)(-nos)")

pipeline = Pipeline().setStages([documentAssembler, tokenizer]).fit(data)

result = pipeline.transform(data)

result.selectExpr("token.result").show(truncate=False)


tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token") \
    .setSuffixPattern("([^\s\w]?)('hi|ls|'l|'ns|'t|'m|'n|-n|-en|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)\z")\
    .setInfixPatterns(["([^\s\w]?)(-nos)"]) \
    .setExceptions(ex_list_all)#.fit(data)


pipeline = Pipeline().setStages([documentAssembler, tokenizer]).fit(data)

result = pipeline.transform(data)

result.selectExpr("token.result").show(truncate=False)