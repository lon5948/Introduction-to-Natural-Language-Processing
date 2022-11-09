import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")

data_path = "dataset.csv"
data = pd.read_csv(data_path, header = None, names=['index','sentence','subject','verb','object'])

def get_sth(sentence, label):
    list = []
    sentence = nlp(sentence)
    for token in sentence:
        if token.pos_ == label:
            list.append(token.text)
            return list

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
        if word.strip() in sen:
            return True
    return False


ans = pd.DataFrame(columns=["index","T/F"])
for row in range(len(data)):
    sen = data["sentence"][row]
    sen = nlp(sen)
    s = []
    v = []
    o = []
    for idx in verb_idxs(sen):
        subj = get_phrase(sen, idx[0],'subj')
        obj = get_phrase(sen, idx[0],'obj')
        verb = sen[idx[0]]
        if subj != None:
            for sub in subj:
                s.append(sub.text)
        if obj != None:
            for ob in obj:
                o.append(ob.text)
        if verb != None:
            v.append(verb.text)

    if get_sth(sen,'AUX')!=None:
        for word in get_sth(sen,'AUX'):
            o.append(word)
    
    a = word_in_sen(s, str(data["subject"][row]))
    b = word_in_sen(v, str(data["verb"][row]))
    c = word_in_sen(o, str(data["object"][row]))
    predict=int(a and b and c)
    ans.loc[row] = [row,predict]

ans.to_csv("123predict.csv",index=False)






