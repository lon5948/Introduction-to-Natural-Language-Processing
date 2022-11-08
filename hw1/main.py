import numpy as np
import pandas as pd
import spacy
from tqdm import trange

nlp = spacy.load("en_core_web_sm")

SUBJECTTAG = {"agent", "expl"}  
OBJECTTAG = {"attr", "oprd"}  

def get_subject_phrase(doc, prevID, verbID):
    subret = []
    for token in doc[prevID+1:verbID]:
        if token.dep_ in SUBJECTTAG or 'subj' in token.dep_:
            subret.append(token)
    if len(subret) != 0:
        subret_conjunction = []
        for sr in subret:
            subret_conjunction.extend(sr.conjuncts)
        subret.extend(subret_conjunction)
    return subret

def get_object_phrase(doc, verbID):
    objret = []
    for token in doc[verbID+1:]:
        if token.dep_ in OBJECTTAG or 'obj' in token.dep_:
            objret.append(token)
        elif token.pos_ == 'VERB':
            break
    if  len(objret) != 0:
        objret_conjunction = []
        for obr in objret:
            objret_conjunction.extend(obr.conjuncts) 
        objret.extend(objret_conjunction)
    return objret

def find_verbs(doc):
    verbs = []
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
        doc = [d for d in doc]
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
    ans.to_csv("submit.csv",index=False)

if __name__ == '__main__':
    main()