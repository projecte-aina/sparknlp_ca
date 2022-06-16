#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:12:32 2022

@author: crodrig1
"""

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
import pandas as pd


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



embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_roberta_base_bne","es") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")


embeddingsSentence = SentenceEmbeddings() \
    .setInputCols(["sentence", "embeddings"]) \
    .setOutputCol("sentence_embeddings") \
    .setPoolingStrategy("AVERAGE")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token") 

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("form")\
    .setLowercase(True)\
    .setCleanupPatterns(["\n "])

stop_words = StopWordsCleaner.pretrained("stopwords_iso","es") \
    .setInputCols(["token"]) \
    .setOutputCol("cleanTokens")
    
lemmatizer = LemmatizerModel.pretrained("lemma_ancora", "es").setInputCols(["sentence", "token"]).setOutputCol("lemma")

pos = PerceptronModel.pretrained("pos_ancora", "es")\
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")


ner = RoBertaForTokenClassification.pretrained("roberta_ner_roberta_base_bne_capitel_ner","es") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")


nerconverter = NerConverter()\
    .setInputCols(["document", "token", "ner"]) \
    .setOutputCol("entities")#\
    #.setWhiteList(['ORG','LOC','PER','MISC'])#\

chunker = Chunker() \
   .setInputCols(["sentence", "pos"]) \
   .setOutputCol("chunk") \
   .setRegexParsers(["<DET>*<NOUN><NOUN>+", "<PROPN>+"])
   

nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentencerDL,
    tokenizer,
    normalizer,
    stop_words,
    embeddings,
    embeddingsSentence,
    embeddingsFinisher,
    lemmatizer,
    pos,
    ner,
    nerconverter,
    chunker
 ])



#Aplicacion de los pipelines y visualización de los resultados
text = "Veo al hombre de los Estados Unidos con el  telescopio"

spark_df = spark.createDataFrame([[text]]).toDF("text")

#doc_df = documentAssembler.transform(spark_df)


empty_df = spark.createDataFrame([['']]).toDF("text")
pipelineModel = nlpPipeline.fit(empty_df)


result = pipelineModel.transform(spark_df)

result.selectExpr("pos.result").show(truncate=False)

from sparknlp.base import LightPipeline

light_model = LightPipeline(pipelineModel)
text = "La Reserva Federal de el Gobierno de EE UU aprueba una de las mayorores subidas de tipos de interés desde 1994."
light_result = light_model.annotate(text)


#print(list(zip(light_result['token'], light_result['pos'], light_result['ner'], light_result['entities'], light_result['chunk'])))


result = pd.DataFrame(zip(light_result['token'], light_result['lemma'], light_result['pos'], light_result['ner']), columns = ["token", "lemma", "pos", "ner"])

print(result)
print("entites:", light_result['entities'])
print("chunk:", light_result['chunk'])
