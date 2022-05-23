#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:27:14 2022

@author: crodrig1
"""

import sparknlp

spark = sparknlp.start()

print("Spark NLP version: ", sparknlp.version())
print("Apache Spark version: ", spark.version)



from sparknlp.base import *

documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")\
      .setCleanupMode("shrink_full")

from sparknlp.annotator import *

# we feed the document column coming from Document Assembler

sentenceDetector = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentences')



sentencerDL = SentenceDetectorDLModel\
  .pretrained("sentence_detector_dl", "xx") \
  .setInputCols(["document"]) \
  .setOutputCol("sentences")
  


retokenizer = RecursiveTokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token") \
    .setPrefixes(["’", '”', "(", "[", "l'","s'","d'","m'","L'","S'","N'","M'"]) \
    .setWhitelist(["aprox.","pàg.","p.ex.","gen.","feb.","abr.","jul.","set.","oct.","nov.","dec.","Dr.","Dra.","Sr.","Sra.","Srta.","núm","St.","Sta.","pl.","etc."] ) \
    .setSuffixes(["-ho","'ls","'l","'ns","'t","'m","'n","’ls","’l","’ns","’t","’m","’n","-les","-la","-lo","-li","-los","-me","-nos","-te","-vos","-se","-hi","-ne","-en",'.', ':', '%', ',', ';', '?', "'", '"', ')', ']', '!'])



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
    .setDictionary("/home/crodrig1/sparknlp/lemma_data/ca_lemma_dict.tsv", "\t", " ")

pos = PerceptronModel.pretrained("pos_ud_ancora", "ca") \
  .setInputCols(["document", "token"]) \
  .setOutputCol("pos")


#ner = RoBertaForTokenClassification.loadSavedModel("/home/crodrig1/sparknlp/ner/projecte-aina/roberta-base-ca-cased-ner_spark_nlp"

ner = RoBertaForTokenClassification\
  .loadSavedModel('./ner/projecte-aina/roberta-base-ca-cased-ner/saved_model/1'.format("roberta-base-ca-cased-ner"), spark)\
  .setInputCols(["document",'token'])\
  .setOutputCol("ner")\
  .setCaseSensitive(True)\
  .setMaxSentenceLength(128)



nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentencerDL,
    retokenizer,
    normalizer,
    lemmatizer,
    pos,
    ner
 ])




text = """Veig a l'home dels Estats Units amb el telescopi.
S'hauria d'afinar la proposta. (algú hauria de fer-ho.) parlem-ne demà, si vols, amb la Dra. Fernandez el 12/25/2022
 Vés-te’n, anem-nos-en, doncs. """
spark_df = spark.createDataFrame([[text]]).toDF("text")

doc_df = documentAssembler.transform(spark_df)


empty_df = spark.createDataFrame([['']]).toDF("text")
pipelineModel = nlpPipeline.fit(empty_df)


result = pipelineModel.transform(spark_df)



import pyspark.sql.functions as F
result_df = result.select(F.explode(F.arrays_zip(result.token.result, result.form.result, result.lemma.result, result.pos.result, result.ner.result)).alias("cols")) \
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("form"),
                          F.expr("cols['2']").alias("lemma"),
                          F.expr("cols['3']").alias("pos"),
                          F.expr("cols['4']").alias("ner") \
                              ).toPandas()

result_df.head(55)





