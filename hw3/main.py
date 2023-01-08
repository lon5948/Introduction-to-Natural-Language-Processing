import json
import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW
from torch.utils.data import Dataset, DataLoader
import torch

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def compute_accuracy(y_pred, y_target):
    outputs = np.argmax(y_pred, axis=1)
    return np.sum(outputs==y_target)

def load_data():
    D = [[], [], []]
    for sid in range(3):
        with open("data/"+["train_HW3dataset.json", "dev_HW3dataset.json", "test_HW3dataset.json"][sid], "r", encoding="utf8") as f:
            data = json.load(f)
        if sid == 0:
            random.shuffle(data)
        for i in range(len(data)):
            for j in range(len(data[i][1])):
                d = ['\n'.join(data[i][0]), data[i][1][j]["question"]]
                for k in range(len(data[i][1][j]["choice"])):
                    d += [data[i][1][j]["choice"][k]]
                for k in range(len(data[i][1][j]["choice"]), 4):
                    d += ['']
                if sid == 2:
                    d += [data[i][1][j]["choice"][0]] 
                else:
                    d += [data[i][1][j]["answer"]] 
                D[sid] += [d]
    return D

def create_examples(data):
    examples = []
    answer = 0
    for i, d in enumerate(data):
        for k in range(4):
            if d[k+2] == d[6]:
                answer = k
        for k in range(4):
            article = d[0]
            question = d[1]
            choice = d[k+2]
            examples.append([article, question, choice, answer])
    return examples

class TaskDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        example = self.examples[index]
        article = example[0]
        question = example[1]
        choice = example[2]
        second_sentences = question + "[SEP]" + choice
        tokenized_examples = self.tokenizer(article, second_sentences, truncation='longest_first', padding='max_length', max_length=512)
        ret_dict = {'input_ids':tokenized_examples['input_ids'],
                    'token_type_ids':tokenized_examples['token_type_ids'],
                    'attention_mask':tokenized_examples['attention_mask'],
                    'label':example[3]}
        return ret_dict

    def __len__(self):
        return len(self.examples)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_data()
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    config = BertConfig().from_pretrained('bert-base-chinese', num_labels = 4)
    config.vocab_size = tokenizer.vocab_size
    
    # training part
    train_examples = create_examples(dataset[0])
    train_data = TaskDataset(train_examples, tokenizer)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config).to(device)
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_parameters, lr=5e-5)
    
    epochs = 1
    print("########## Running Training ###########")
    for epoch in range(epochs):
        print(f'epoch {epoch+1}')
        model.train()
        total_acc = 0
        total_loss = 0
        best_acc = 0
        count = 0
        for batch_dict in tqdm(train_loader):
            input_ids = torch.stack(batch_dict['input_ids'],dim=1).to(device)
            token_type_ids = torch.stack(batch_dict['token_type_ids'],dim=1).to(device)
            attention_mask = torch.stack(batch_dict['attention_mask'],dim=1).to(device)
            labels = torch.as_tensor(batch_dict['label'], dtype=torch.long).to(device)
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs[:2]
            loss = loss.to(torch.float32)
            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            acc = compute_accuracy(logits, labels)
            total_acc += acc
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += input_ids.size(0)
        total_acc /= count
        total_loss /= count
        print(f'Loss = {total_loss}')
        print(f'Accuracy = {total_acc}')
        if total_acc >= best_acc:
            torch.save(model.state_dict(), "model_best.pt")
            best_acc = total_acc
    
    # dev part
    dev_examples = create_examples(dataset[1])
    dev_data = TaskDataset(dev_examples, tokenizer)
    dev_loader = DataLoader(dev_data, batch_size=4)

    model.load_state_dict(torch.load("model_best.pt"))
    model.eval()
    total_dev_acc = 0
    total_dev_loss = 0
    count = 0
    print("########## Running Evaluation ###########")
    for dev_dict in tqdm(dev_loader):
        dev_input_ids = torch.stack(dev_dict['input_ids'],dim=1).to(device)
        dev_token_type_ids = torch.stack(dev_dict['token_type_ids'],dim=1).to(device)
        dev_attention_mask = torch.stack(dev_dict['attention_mask'],dim=1).to(device)
        dev_labels = torch.as_tensor(dev_dict['label'], dtype=torch.long).to(device)
        dev_outputs = model(input_ids=dev_input_ids, token_type_ids=dev_token_type_ids, attention_mask=dev_attention_mask, labels=dev_labels)
        dev_loss, dev_logits = dev_outputs[:2]
        dev_logits = dev_logits.detach().cpu().numpy()
        dev_labels = dev_labels.to('cpu').numpy()
        dev_acc = compute_accuracy(dev_logits, dev_labels)
        total_dev_acc += dev_acc
        total_dev_loss += dev_loss
        count += dev_input_ids.size(0)
    total_dev_acc /= count
    total_dev_loss /= count
    print(f'Loss = {total_loss}')
    print(f'Accuracy = {total_acc}')
    
    # testing part
    ans = pd.DataFrame(columns=["index","answer"])
    ans_index = 0

    test_examples = create_examples(dataset[2])
    test_data = TaskDataset(test_examples, tokenizer)
    test_loader = DataLoader(test_data, batch_size=2)

    model.load_state_dict(torch.load("model_best.pt"))
    model.eval()
    print("########## Running Testing ###########")
    for test_dict in tqdm(test_loader):
        test_input_ids = torch.stack(test_dict['input_ids'],dim=1).to(device)
        test_token_type_ids = torch.stack(test_dict['token_type_ids'],dim=1).to(device)
        test_attention_mask = torch.stack(test_dict['attention_mask'],dim=1).to(device)
        output = model(input_ids=test_input_ids, token_type_ids=test_token_type_ids, attention_mask=test_attention_mask)
        logits = output[0]
        logits = logits.detach().cpu().numpy()
        for i in range(len(logits)):
            ans.loc[int(ans_index)] = [int(ans_index),np.argmax(logits[i])+1]
            ans_index += 1
    ans.to_csv('predict.csv', index=False)
if __name__ == "__main__":
    main()