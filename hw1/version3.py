#Author: Ming-Rung Li
#Student ID: 109550031
#HW ID: HW1

import numpy as np
import pandas as pd
import spacy
from tqdm import trange

nlp = spacy.load("en_core_web_sm")

SUBJECTTAG = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]  
OBJECTTAG = ["dobj", "pobj", "dative", "attr", "oprd"]  

def add_conjunction(ret):
    conjunction = []
    for r in ret:
        conjunction.extend(r.conjuncts)
    ret.extend(conjunction)
    return ret

def get_subject_phrase(doc, prevID, verbID):
    subret = []
    doc = nlp(doc)
    for token in doc[prevID+1:verbID]:
        if token.dep_ in SUBJECTTAG:
            subret.append(token)
    if len(subret) != 0:
        subret = add_conjunction(subret)
    return subret

def get_object_phrase(doc, verbID):
    objret = []
    doc = nlp(doc)
    for token in doc[verbID+1:]:
        if token.dep_ in OBJECTTAG:
            objret.append(token)
        elif token.pos_ == 'VERB':
            break
    if len(objret) != 0:
        objret = add_conjunction(objret)
    return objret

def find_verbs(doc):
    verbs = []
    doc = nlp(doc)
    for token in doc:
        if token.pos_ == 'VERB' or token.pos_ == 'AUX':
            verbs.append(token)
    return verbs

def word_in_sen(s,sen):
    for word in s:
        if word in sen:
            return True
    return False

def main():
    data_path = "dataset.csv"
    data= pd.read_csv(data_path,header=None,names=['index','sentence','subject','verb','object'])
    ans = pd.DataFrame(columns=["index","T/F"])

    for row in trange(len(data)):
        sen = data["sentence"][row]
        doc = nlp(sen)
        s = []
        v = []
        o = []
        prevID = -1
        verbID = 0
        for verb in find_verbs(doc):
            verbID = verb.i
            subj = get_subject_phrase(doc, prevID, verbID)
            obj = get_object_phrase(doc, verbID)
            if subj != None:
                for sub in subj:
                    s.append(sub.text)
            if obj != None:
                for ob in obj:
                    o.append(ob.text)
            if verb != None:
                v.append(verb.text)
            prevID = verb.i
        
        a = word_in_sen(s, str(data["subject"][row]))
        b = word_in_sen(v, str(data["verb"][row]))
        c = word_in_sen(o, str(data["object"][row]))
        predict = int(a and b and c)
        ans.loc[row] = [row,predict]
    ans.to_csv("predict.csv",index=False)

if __name__ == '__main__':
    main()