# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sparknlp.training import CoNLLU
from sparknlp.base import *
from sparknlp.annotator import *
import sparknlp
spark = sparknlp.start()
# ancora_path = "/mnt/BSC/catala/corpora/ANCORA_ca/"

# conlluFile = ancora_path+"ca_ancora-ud-train.conllu"
# conllDataSet = CoNLLU().readDataset(spark, conlluFile)
# conllDataSet.selectExpr(
#     "text",
#     "form.result as form",
#     "upos.result as upos",
#     "xpos.result as xpos",
#     "lemma.result as lemma"
# ).show(1, False)


import json

lexis = json.load(open("ca_lemma_lookup.json"))
excl = json.load(open("ca_lemma_exc.json"))



dictiolex = {}

for w in lexis.keys():
    l = lexis[w][0]
    if l in dictiolex.keys():
        words = dictiolex[l]
        dictiolex[l] = list(set(words + [w]))
    else:
        words = [w,l]
        dictiolex[l] = words

for pos in excl.keys():
    posos = excl[pos]
    for w in posos.keys():
        l = posos[w][0]
        if l in dictiolex.keys():
            words = dictiolex[l]
            dictiolex[l] = list(set(words + [w]))
        else:
            words = [w,l]
            dictiolex[l] = words            

with open("ca_lemma_dict.tsv","w") as fp:
    for l in dictiolex.keys():
        fp.write(l+"\t"+" ".join(dictiolex[l])+"\n")