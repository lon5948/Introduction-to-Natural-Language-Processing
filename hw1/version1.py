import numpy as np
import pandas as pd
import spacy
from tqdm import trange

nlp = spacy.load("en_core_web_sm")

SUBJECTTAG = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]  
OBJECTTAG = ["dobj", "dative", "attr", "oprd", "pobj"]  

def get_subject_phrase(doc):
    subret = []
    for token in doc:
        if token.dep_ in SUBJECTTAG:
            subret.append(token)
    return subret

def get_object_phrase(doc):
    objret = []
    for token in doc:
        if token.dep_ in OBJECTTAG:
            objret.append(token)
    return objret

def find_verbs(doc):
    verbs = []
    for token in doc:
        if token.pos_ == 'VERB':
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
        
        verb = find_verbs(doc)
        subj = get_subject_phrase(doc)
        obj = get_object_phrase(doc)
            
        if subj != None:
            for sub in subj:
                s.append(sub.text)
        if obj != None:
            for ob in obj:
                o.append(ob.text)
        if verb != None:
            for ve in verb:
                v.append(ve.text)
        
        a = word_in_sen(s, str(data["subject"][row]))
        b = word_in_sen(v, str(data["verb"][row]))
        c = word_in_sen(o, str(data["object"][row]))
        predict = int(a and b and c)
        ans.loc[row] = [row,predict]
    ans.to_csv("version1_predict.csv",index=False)

if __name__ == '__main__':
    main()