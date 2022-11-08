import pandas as pd
import spacy
from tqdm import trange
nlp=spacy.load("en_core_web_sm")
'''
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
'''
data_path="dataset.csv"
data=pd.read_csv(data_path,header=None,names=['index','sentence','subject','verb','object'])

def get_sth(sen,label):
    list=[]
    sen=nlp(sen)
    for token in sen:
        if token.pos_==label:
            list.append(token.text)
            return list


#Try to improve this function
def get_phrase(sen,head_idx,tag):
    sen=nlp(sen)
    for token in sen:
        if tag in token.dep_ and token.head.i==head_idx:
            subtree=list(token.subtree)
            start=subtree[0].i
            end=subtree[-1].i+1
            return sen[start:end]

def verb_idxs(sen):
    sen=nlp(sen)
    idxs=[(i,token) for i,token in enumerate(sen)if token.pos_=='VERB']
    return idxs

def word_in_sen(s,sen):
    for word in s:
        if word in sen:
            return True
    return False
ans=pd.DataFrame(columns=["index","T/F"])


for row in trange(len(data)):
    sen=data["sentence"][row]
    sen=nlp(sen)
    s = []
    v = []
    o = []
    for idx in verb_idxs(sen):
        subj=get_phrase(sen,idx[0],'subj')
        obj=get_phrase(sen,idx[0],'obj')
        verb = sen[idx[0]]
        if subj != None:
            s.append(subj.text)
        if obj != None:
            o.append(obj.text)
        if verb != None:
            v.append(verb.text)
# Maybe consider the other Part-of-Speech?
# Ex:
    '''
    if get_sth(sen,'AUX')!=None:
        for word in get_sth(sen,'AUX'):
            o.append(word)
    '''

    a = word_in_sen(s, str(data["subject"][row]))
    b = word_in_sen(v, str(data["verb"][row]))
    c = word_in_sen(o, str(data["object"][row]))
    predict=int(a and b and c)

    ans.loc[row] = [row,predict]

ans.to_csv("predict.csv",index=False)






