#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:27:14 2022

@author: crodrig1

Spark NLP version:  3.4.4
Apache Spark version:  3.1.2
"""

import sparknlp

spark = sparknlp.start()#spark32=True)

print("Spark NLP version: ", sparknlp.version())
print("Apache Spark version: ", spark.version)

#Spark NLP version:  3.4.4
#Apache Spark version:  3.1.2

from sparknlp.annotator import *
from sparknlp.base import *

documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")\
      .setCleanupMode("shrink_full")


sentencerDL = SentenceDetectorDLModel\
  .pretrained("sentence_detector_dl", "xx") \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")
  

embeddings = RoBertaEmbeddings.load("/home/crodrig1/sparknlp/sparknlp_ca/PlanTL-GOB-ES/roberta-base-ca_spark_nlp".format('roberta-base'))\
  .setInputCols(["sentence",'token'])\
  .setOutputCol("embeddings")\
  .setCaseSensitive(True)

# retokenizer = RecursiveTokenizer() \
#     .setInputCols(["document"]) \
#     .setOutputCol("token") \
#     .setPrefixes(["’", '”', "(", "[", "l'","l’","s'","s’","d’","d'","m’","m'","L'","L’","S’","S'","N’","N'","M’","M'"]) \
#     .setWhitelist(["aprox.","pàg.","p.ex.","gen.","feb.","abr.","jul.","set.","oct.","nov.","dec.","Dr.","Dra.","Sr.","Sra.","Srta.","núm","St.","Sta.","pl.","etc."] ) \
#     .setSuffixes(["-ho","'ls","'l","'ns","'t","'m","'n","’ls","’l","’ns","’t","’m","’n","-les","-la","-lo","-li","-los","-me","-nos","-te","-vos","-se","-hi","-ne","-en",'.', ':', '%', ',', ';', '?', "'", '"', ')', ']', '!'])

tokenizer = Tokenizer()# \
    # .setInputCols(['sentence']) \
    # .setOutputCol('token')\
    # .setPrefixes(["’", '”', "(", "[", "l'","l’","s'","s’","d’","d'","m’","m'","L'","L’","S’","S'","N’","N'","M’","M'"]) \
    # .setWhitelist(["aprox.","pàg.","p.ex.","gen.","feb.","abr.","jul.","set.","oct.","nov.","dec.","Dr.","Dra.","Sr.","Sra.","Srta.","núm","St.","Sta.","pl.","etc."] ) \
    # .setSuffixes(["-ho","'ls","'l","'ns","'t","'m","'n","’ls","’l","’ns","’t","’m","’n","-les","-la","-lo","-li","-los","-me","-nos","-te","-vos","-se","-hi","-ne","-en",'.', ':', '%', ',', ';', '?', "'", '"', ')', ']', '!'])


# lemmatizer = LemmatizerModel.pretrained("lemma_spacylookup","ca") \
#     .setInputCols(["form"]) \
#     .setOutputCol("lemma")

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("form")\
    .setLowercase(True)\
    .setCleanupPatterns(["\n "])

lemmatizer = Lemmatizer() \
    .setInputCols(["form"]) \
    .setOutputCol("lemma") \
    .setDictionary("/home/crodrig1/sparknlp/sparknlp_ca/ca_lemma_dict.tsv", "\t", " ")

pos = PerceptronModel.pretrained("pos_ud_ancora", "ca") \
  .setInputCols(["document", "token"]) \
  .setOutputCol("pos")


#Per cargar per primera vegada despres de la conversió des de PyTorch 
# ner = RoBertaForTokenClassification\
#   .loadSavedModel('/home/crodrig1/sparknlp/sparknlp_ca/projecte-aina/roberta-base-ca-cased-ner/saved_model/1'.format("roberta-base-ca-cased-ner"), spark)\
#   .setInputCols(["document",'token'])\
#   .setOutputCol("ner")\
#   .setCaseSensitive(True)\
#   .setMaxSentenceLength(128)

# Despres de save()
# wget and unzip: 
# https://github.com/projecte-aina/sparknlp_ca/releases/download/NER_v2/roberta-base-ca-cased-ner_spark_nlp.zip
ner = RoBertaForTokenClassification.load("/home/crodrig1/sparknlp/sparknlp_ca/roberta-base-ca-cased-ner_spark_nlp")
ner.setOutputCol('ner')


nerconverter = NerConverter()\
    .setInputCols(["document", "token", "ner"]) \
    .setOutputCol("entities")#\
    #.setWhiteList(['ORG','LOC','PER','MISC'])#\


nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentencerDL,
    tokenizer,
    embeddings,
    normalizer,
    lemmatizer,
    pos,
    ner,
    nerconverter
 ])



#Aplicacion de los pipelines y visualización de los resultados
text = "Veig a l'home dels Estats Units amb el telescopi."

spark_df = spark.createDataFrame([[text]]).toDF("text")

#doc_df = documentAssembler.transform(spark_df)


empty_df = spark.createDataFrame([['']]).toDF("text")
pipelineModel = nlpPipeline.fit(empty_df)


result = pipelineModel.transform(spark_df)

result.select('entities.result').show(truncate=False)

import pyspark.sql.functions as F
result_df = result.select(F.explode(F.arrays_zip(result.token.result, result.form.result, result.lemma.result, result.pos.result,result.ner.result,result.entities.result)).alias("cols")) \
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("form"),
                          F.expr("cols['2']").alias("lemma"),
                          F.expr("cols['3']").alias("pos"),
                          F.expr("cols['4']").alias("ner"),
                          F.expr("cols['5']").alias("entities")\
                              ).toPandas()

print(result_df.head(20))

from sparknlp.base import LightPipeline

light_model = LightPipeline(pipelineModel)

light_result = light_model.annotate("La sala del contenciós-administratiu del Tribunal Suprem espanyol ha rectificat i ha anunciat ara que revisarà l’indult als presos polítics concedits pel govern espanyol, en tant que tramitarà els recursos interposats pel PP, Ciutadans i Vox en contra.")



print(list(zip(light_result['token'], light_result['lemma'], light_result['ner'], light_result['entities'])))


