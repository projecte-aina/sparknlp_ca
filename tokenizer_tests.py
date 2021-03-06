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



#text = """A partir del any 2020, l'incidència delicada als virus "Nocius" (o 'nefandos'), s'hauria d'abaixar, fent-la passar del 21,1 per cent l'any 1999 al 19,4 (per cent de 2000). Parlem-ne."""
text = """La hipòtesi que avança l'Elisi i que sacsegen frenèticament els cacics de d'jean-marc és la del complot (socialista). Cal fer-les amb il·lusió, i durant el 10%, 1990, la Sagrada Família "mostrarà una imatge aclaparadora". """
data = spark.createDataFrame([[text]]).toDF("text")

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
  .pretrained("sentence_detector_dl", "xx") \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")

ex_list = ["aprox\.","pàg\.","p\.ex\.","gen\.","feb\.","abr\.","jul\.","set\.","oct\.","nov\.","dec\.","dr\.","dra\.","sr\.","sra\.","srta\.","núm\.","st\.","sta\.","pl\.","etc\.", "ex\."]#,"’", '”', "(", "[", "l'","l’","s'","s’","d’","d'","m’","m'","L'","L’","S’","S'","N’","N'","M’","M'"]
ex_list_all = []
ex_list_all.extend(ex_list)
ex_list_all.extend([x[0].upper() + x[1:] for x in ex_list])
ex_list_all.extend([x.upper() for x in ex_list])

tokenizer = Tokenizer() \
     .setInputCols(['sentence']).setOutputCol('token')\
     .setInfixPatterns(["(d|D)(els)","(d|D)(el)","(a|A)(ls)","(a|A)(l)","(p|P)(els)","(p|P)(el)",\
                            "([A-zÀ-ú_@]+)(-[A-zÀ-ú_@]+)",\
                             "(d'|D')([·A-zÀ-ú@_]+)(\.|\"|;|:|!|\?|\-|\(|\)|”|“|'|,)+","(l'|L')([·A-zÀ-ú_]+)(\.|\"|;|:|!|\?|\-|\(|\)|”|“|'|,)+", \
                             "(l'|l'|s'|s'|d'|d'|m'|m'|n'|n'|D'|D'|L'|L'|S'|S'|N'|N'|M'|M')([A-zÀ-ú_]+)",\
                             """([A-zÀ-ú·]+)(\.|,|\)|\?|!|;|\:|\"|”)(\.|,|\)|\?|!|;|\:|\"|”)+""",\
                             "([A-zÀ-ú·]+)('l|'ns|'t|'m|'n|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)(\.|,|;|:|\?|,)+",\
                             "([A-zÀ-ú·]+)('l|'ns|'t|'m|'n|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)",\
                             "(\.|\"|;|:|!|\?|\-|\(|\)|”|“|')+([0-9A-zÀ-ú_]+)",\
                             "([0-9A-zÀ-ú·]+)(\.|\"|;|:|!|\?|\(|\)|”|“|'|,|%)",\
                             "(\.|\"|;|:|!|\?|\-|\(|\)|”|“|,)+([0-9]+)(\.|\"|;|:|!|\?|\-|\(|\)|”|“|,)+",\
                             "(d'|D'|l'|L')([·A-zÀ-ú@_]+)('l|'ns|'t|'m|'n|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)(\.|\"|;|:|!|\?|\-|\(|\)|”|“|,)", \
                             "([\.|\"|;|:|!|\?|\-|\(|\)|”|“|,]+)([\.|\"|;|:|!|\?|\-|\(|\)|”|“|,]+)"]) \
         .setExceptions(ex_list_all).fit(data)


pipeline = Pipeline().setStages([documentAssembler, sentencerDL,  tokenizer]).fit(data)
light_model = LightPipeline(pipeline)


light_result = light_model.annotate(text)

import pandas as pd

result = pd.DataFrame(zip(light_result['token']), columns = ["token"])

print(result)


