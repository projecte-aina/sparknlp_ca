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



data = spark.createDataFrame([["A partir d'aquest any, la incidència ha anat baixant, passant del 21,1 per cent l'any 1999 al 19,4 per cent de 2000."]]).toDF("text")

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
                             """([A-zÀ-ú·]+)(\.|,|")""",\
                             "([A-zÀ-ú·]+)('l|'ns|'t|'m|'n|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)(\.|,|;|:|\?|,)+",\
                             "([A-zÀ-ú·]+)('l|'ns|'t|'m|'n|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)",\
                             "(\.|\"|;|:|!|\?|\-|\(|\)|”|“|')+([0-9A-zÀ-ú_]+)",\
                             "([0-9A-zÀ-ú]+)(\.|\"|;|:|!|\?|\(|\)|”|“|'|,)+",\
                             "(\.|\"|;|:|!|\?|\-|\(|\)|”|“|,)+([0-9]+)(\.|\"|;|:|!|\?|\-|\(|\)|”|“|,)+",\
                             "(d'|D'|l'|L')([·A-zÀ-ú@_]+)('l|'ns|'t|'m|'n|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)(\.|\"|;|:|!|\?|\-|\(|\)|”|“|,)", \
                             """([\.|"|;|:|!|\?|\-|\(|\)|”|“|,]+)([\.|"|;|:|!|\?|\-|\(|\)|”|“|,]+)"""]) \
         .setExceptions(ex_list_all).fit(data)


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

ner = RoBertaForTokenClassification.load("./roberta-base-ca-cased-ner_spark_nlp")
ner.setOutputCol('ner')


nerconverter = NerConverter()\
    .setInputCols(["document", "token", "ner"]) \
    .setOutputCol("entities")#\

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

text = "A partir d'aquest any, la incidència ha anat baixant, passant del 21,1 per cent l'any 1999 al 19,4 per cent de 2000."#"venien (del delta) a buscar l'aigua. anem-nos-en de la casa. ella."
spark_df = spark.createDataFrame([[text]]).toDF("text")

#doc_df = documentAssembler.transform(spark_df)


empty_df = spark.createDataFrame([['']]).toDF("text")
pipelineModel = nlpPipeline.fit(empty_df)

#pipelineModel.save("/home/crodrig1/sparknlp/pipeline_md_ca")

result = pipelineModel.transform(spark_df)


from sparknlp.base import LightPipeline

light_model = LightPipeline(pipelineModel)


light_result = light_model.annotate(text)


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
            try:
                if referencet[n] != testt[n]:
                    print(referencet[n],testt[n])
                    return None
            except IndexError:
                return None
    return score

token = []
pos = []
ner = []
lemma = []
tokerrors = []
malprocessed = []
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
        try:
            if referencet[n] != testt[n]:
                tokerrors.append((referencet[n],testt[n]))
                break
        except IndexError:
            malprocessed.append(sentence)
    score = measure("token",dtok,rd)
    if score:
        token.append(score)
        pos.append(measure("pos",dtok,rd))
        ner.append(measure("ner",dtok,rd))
        lemma.append(measure("lemma",dtok,rd))
    else:
        m += 1
    #list(dtok['pos'].values())
from statistics import mean

   
print("Total phrases to process: ", len(list(newdata.keys())))
print("Correctly tokenized ",len(token))
print("Incorrectly tokenized at least on token: ", m)
print("Percent correct: ",(len(token)+100)/len(list(newdata.keys())))


print("average pos: ",mean(pos))
print("average ner: ",mean(ner))
print("average lemma: ",mean(lemma))


from collections import Counter
tokenerr = Counter(tokerrors)
print("Common errors in tokenization")
print(tokenerr.most_common(20))
with open("tok_errors.tsv","w") as fp:
    for err,num in tokenerr.most_common():
        fp.write(err[0]+"\t"+err[1]+"\t"+str(num)+"\n")
fp.close()
