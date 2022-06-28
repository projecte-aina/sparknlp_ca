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

# sentembeddings = RoBertaSentenceEmbeddings.load("/home/crodrig1/sparknlp/sparknlp_ca/roberta-base-ca_spark_nlp").format('roberta_base')\
#   .setInputCols(["sentence"])\
#   .setOutputCol("sentence_embeddings")


embeddings = RoBertaEmbeddings.load("./roberta-base-ca_spark_nlp")\
  .setInputCols(["sentence",'token'])\
  .setOutputCol("embeddings")\
  .setCaseSensitive(True)


embeddingsSentence = SentenceEmbeddings() \
    .setInputCols(["sentence", "embeddings"]) \
    .setOutputCol("sentence_embeddings") \
    .setPoolingStrategy("AVERAGE")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)
# retokenizer = RecursiveTokenizer() \
#     .setInputCols(["document"]) \
#     .setOutputCol("token") \
#     .setPrefixes(["’", '”', "(", "[", "l'","l’","s'","s’","d’","d'","m’","m'","L'","L’","S’","S'","N’","N'","M’","M'"]) \
#     .setWhitelist(["aprox.","pàg.","p.ex.","gen.","feb.","abr.","jul.","set.","oct.","nov.","dec.","Dr.","Dra.","Sr.","Sra.","Srta.","núm","St.","Sta.","pl.","etc."] ) \
#     .setSuffixes(["-ho","'ls","'l","'ns","'t","'m","'n","’ls","’l","’ns","’t","’m","’n","-les","-la","-lo","-li","-los","-me","-nos","-te","-vos","-se","-hi","-ne","-en",'.', ':', '%', ',', ';', '?', "'", '"', ')', ']', '!'])




ex_list = ["aprox.","pàg.","p.ex.","gen.","feb.","abr.","jul.","set.","oct.","nov.","des.","dr.","dra.","sr.","sra.","srta.","núm.","st.","sta.","pl.","etc.", "ex."]
#,"’", '”', "(", "[", "l'","l’","s'","s’","d’","d'","m’","m'","L'","L’","S’","S'","N’","N'","M’","M'"]
ex_list_all = []
ex_list_all.extend(ex_list)
ex_list_all.extend([x[0].upper() + x[1:] for x in ex_list])
ex_list_all.extend([x.upper() for x in ex_list])

# tokenizer = Tokenizer() \
#     .setInputCols(['sentence']).setOutputCol('token') \
#     .setContextChars(['.', ',', ';', ':', '!', '?', '*', '-', '(', ')', '"', "'", "«", "»"])\
#     .setSuffixPattern("([A-zÀ-ú]*)(-la|-lo|-les|-los|-hi|-en|-ho|'n|'l|'ls|'m|'t|hi|ho|-LA|-LO|-LES|-LOS|-HI|-EN|-HO|'N|'L|'LS|'M|'T|HI|HO|)(.|,|;|:|!|\?|\)|\", »|)\z")\
#     .setInfixPatterns(["(\"|«|¿|\(|^)(d'|l'|D'|L')([A-zÀ-ú]*)", "(\"|«|¿|\(|^)(d|p|D|P)(el|els|EL|ELS)$", "(\"|«|¿|\(|^)(a|A)(l|ls|L|LS)$", "([A-zÀ-ú]*)(-la|-lo|-les|-los|-nos|-vos|-te|-hi|-en|-ho|-n'|-l'|'ls|-m'|-t'|-hi|-ho|-LA|-LO|-LES|-LOS|-NOS|-VOS|-TE|-HI|-EN|-HO|-N'|-L'|'LS|-M'|-T'|-HI|-HO|)"]) \
#     .setExceptions(ex_list_all)
# ex_list = ["aprox.","pàg.","p.ex.","gen.","feb.","abr.","jul.","set.","oct.","nov.","dec.","dr.","dra.","sr.","sra.","srta.","núm.","st.","sta.","pl.","etc.", "ex."]#,"’", '”', "(", "[", "l'","l’","s'","s’","d’","d'","m’","m'","L'","L’","S’","S'","N’","N'","M’","M'"]
# ex_list_all = []
# ex_list_all.extend(ex_list)
# ex_list_all.extend([x[0].upper() + x[1:] for x in ex_list])
# ex_list_all.extend([x.upper() for x in ex_list])
#print(">>>>>>", ex_list_all)
data = spark.createDataFrame([["A partir d'aquest any, la incidència ha anat baixant, passant del 21,1 per cent l'any 1999 al 19,4 per cent de 2000."]]).toDF("text")
# tokenizer = Tokenizer() \
#     .setInputCols(['sentence']).setOutputCol('token') \
#     .setSuffixPattern("(\w*)(-la|-lo|-les|-los|-hi|-en|-ho|'n|'l|'ls|'m|'t|hi|ho|-LA|-LO|-LES|-LOS|-HI|-EN|-HO|'N|'L|'LS|'M|'T|HI|HO|)(.|,|;|:|!|\?|\)|\"|)\z") \
#     .setInfixPatterns(["^(d'|l'|D'|L')(\w*)", "^(d|p|D|P)(el|els|EL|ELS)?$", "^(a|A)(l|ls|L|LS)?$", "(\w*)(-la|-lo|-les|-los|-nos|-vos|-te|-hi|-en|-ho|-n'|-l'|'ls|-m'|-t'|-hi|-ho|-LA|-LO|-LES|-LOS|-NOS|-VOS|-TE|-HI|-EN|-HO|-N'|-L'|'LS|-M'|-T'|-HI|-HO|)"]) \
#     .setContextChars(['.', ',', ';', ':', '!', '?', '*', '-', '(', ')', '"', "'"]) \
#     .setExceptions(ex_list_all)#.fit(data)
tokenizer = Tokenizer() \
    .setInputCols(['sentence']).setOutputCol('token')\
    .setInfixPatterns(["(\w+)(-\w+)", "(l'|l'|s'|s'|d'|d'|m'|m'|L'|L'|S'|S'|N'|N'|M'|M')(\w+)", "(\w+)(\.|,)"])
#tokenizer.setSuffixPattern("([^\s\w]?)([\-hi]*)\z")#"(ls|'l|'ns|'t|'m|'n|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)\z")

##https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.Tokenizer.html


# lemmatizer = LemmatizerModel.pretrained("lemma_spacylookup","ca") \
#     .setInputCols(["form"]) \
#     .setOutputCol("lemma")

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("form")\
    .setLowercase(True)\
    .setCleanupPatterns(["\n "])

stop_words = StopWordsCleaner.pretrained("stopwords_iso","ca") \
    .setInputCols(["token"]) \
    .setOutputCol("cleanTokens")
    
lemmatizer = Lemmatizer() \
    .setInputCols(["form"]) \
    .setOutputCol("lemma") \
    .setDictionary("ca_lemma_dict.tsv", "\t", " ")

pos = PerceptronModel.pretrained("pos_ancora", "ca") \
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
ner = RoBertaForTokenClassification.load("./roberta-base-ca-cased-ner_spark_nlp")
ner.setOutputCol('ner')


nerconverter = NerConverter()\
    .setInputCols(["document", "token", "ner"]) \
    .setOutputCol("entities")#\
    #.setWhiteList(['ORG','LOC','PER','MISC'])#\

# chunker = Chunker() \
#    .setInputCols(["sentence", "pos"]) \
#    .setOutputCol("chunk") \
#    .setRegexParsers(["<DET>*<NOUN>+", "<PROPN>+"])
chunker = Chunker() \
   .setInputCols(["sentence", "pos"]) \
   .setOutputCol("chunk") \
   .setRegexParsers(["<DET>*<ADV>*<ADJ>*<NOUN>+<ADV>*<ADJ>*", "<DET>*<PROPN>+", "<DET>+<ADV>*<ADJ>+<ADV>*", "<PRON>"])   

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

#nlpPipeline.write().overwrite().save("/home/crodrig1/sparknlp/pipeline_ca")

#Aplicacion de los pipelines y visualización de los resultados
#text = "Veig a l'home dels Estats Units amb el telescopi."
text = "A partir d'aquest any, la incidència ha anat baixant, passant del 21,1 per cent l'any 1999 al 19,4 per cent de 2000."#"venien (del delta) a buscar l'aigua. anem-nos-en de la casa. ella."
spark_df = spark.createDataFrame([[text]]).toDF("text")

#doc_df = documentAssembler.transform(spark_df)


empty_df = spark.createDataFrame([['']]).toDF("text")
pipelineModel = nlpPipeline.fit(empty_df)
#pipelineModel.write().overwrite().save("/home/crodrig1/sparknlp/pipeline_ca")

result = pipelineModel.transform(spark_df)

#result.select('entities.result').show(truncate=False)

# print("ShowSentence  Embeddings")
# result.selectExpr("explode(finished_embeddings) as result").show(5, 50)

# import pyspark.sql.functions as F
# result_df = result.select(F.explode(F.arrays_zip(result.token.result, result.form.result, result.lemma.result, result.pos.result,result.ner.result,result.chunk.result,result.entities.result)).alias("cols")) \
#                   .select(F.expr("cols['0']").alias("token"),
#                           F.expr("cols['1']").alias("form"),
#                           F.expr("cols['2']").alias("lemma"),
#                           F.expr("cols['3']").alias("pos"),
#                           F.expr("cols['4']").alias("ner"),
#                           F.expr("cols['5']").alias("chunk"),
#                           F.expr("cols['6']").alias("entities")\
#                               ).toPandas()
# print(result_df)
from sparknlp.base import LightPipeline

light_model = LightPipeline(pipelineModel)


light_result = light_model.annotate(text)



#print(list(zip(light_result['token'], light_result['pos'], light_result['ner'], light_result['entities'], light_result['chunk'])))


result = pd.DataFrame(zip(light_result['token'], light_result['lemma'], light_result['form'], light_result['pos'], light_result['ner']), columns = ["token", "lemma","form", "pos", "ner"])

print(result)


newdata = {}


frases = open("ca_ancora-ud-test.conll").read().split("\n\n")

def dicToks(toks):
    token = {}
    pos = {}
    ner = {}
    lemma = {}
    i = 0
    for t,l,p,n in toks:
        token[i] = t
        lemma[i] = l 
        pos[i] = p
        ner[i] = n
        i += 1
    return {'token':token,'lemma':lemma,'pos':pos,'ner':ner}
    

for each in frases:
    tokens = []
    for line in each.split("\n"):
        if line.startswith("# sent_id = "):
            idx = line[12:]
        elif line.startswith("# text = "):
            s = line[9:]
        elif line.startswith("# orig_file_sentence"):
            pass
        elif line == '':
            pass
        else:
            tokens.append(line.split("\t"))
    newdata[idx] = (s,tokens)
from nltk.metrics.scores import accuracy
def measure(label,dtok,rd):
    score = None
    # if len(dtok[label]) != len(rd[label]):
    #     print("Mismatched Number of Tokens")
    reference = list(dtok[label].values())
    test = list(rd[label].values())
    try:
        score = accuracy(reference, test)
    except ValueError:
        #print("Mismatched Number of Tokens")
        referencet = list(dtok["token"].values())
        testt = list(rd["token"].values())
        for n in range(len(referencet)):
            if referencet[n] != testt[n]:
                print(referencet[n],testt[n])
                return None
    return score

root = "./tests/"
#from nltk.metrics.scores import accuracy
token = []
pos = []
ner = []
lemma = []
tokerrors = []
m = 0
for k in list(newdata.keys()):
    sentence, toks = newdata[k]  
    dtok = dicToks(toks)
    light_result = light_model.annotate(sentence)
    result = pd.DataFrame(zip(light_result['token'], light_result['lemma'], light_result['pos'], light_result['ner']), columns = ["token", "lemma", "pos", "ner"])
    #print(result)
    rd = result.to_dict()
    referencet = list(dtok["token"].values())
    testt = list(rd["token"].values())
    for n in range(len(referencet)):
        if referencet[n] != testt[n]:
            tokerrors.append((referencet[n],testt[n]))
            break
    score = measure("token",dtok,rd)
    if score:
        token.append(score)
    else:
        m += 1
    #list(dtok['pos'].values())
    
from collections import Counter
tokenerr = Counter(tokerrors)
print(tokenerr.most_common(20))

with open("tok_errors.tsv","w") as fp:
    for err,num in tokenerr.most_common():
        fp.write(err[0]+"\t"+err[1]+"\t"+str(num)+"\n")
fp.close()
