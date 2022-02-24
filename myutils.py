# Author: Sina Mahdipour Saravani
# Link to our paper for this project:
#
import sys
import json
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification\
    , BertForPreTraining, AutoModel
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, mean_squared_error
import time
# from vlad.mynextvlad import NeXtVLAD  # this imports our implementation of the NeXtVLAD layer
# from vlad.internetnextvlad import NextVLAD  # this imports an implementation of the NeXtVLAD layer taken from
# https://www.kaggle.com/gibalegg/mtcnn-nextvlad#NextVLAD

MAXTOKENS = 512
BERT_EMB = 768  # set to either 768 or 1024 for BERT-Base and BERT-Large models respectively
CUDA_0 = 'cuda:1'
CUDA_1 = 'cuda:1'
CUDA_2 = 'cuda:1'

# The function for printing in both console and a given log file.
def myprint(mystr, logfile):
    print(mystr)
    print(mystr, file=logfile)


# The function for loading datasets from parallel tsv files and returning texts in lists.
def load_data(file_name):
    try:
        # f = open(file_name)
        f = pd.read_csv(file_name, sep='\t', names=['l1_text', 'l2_text'])#, 'extra'])
    except:
        print('my log: could not read file')
        exit()
    print("This many number of rows were removed from " + file_name.split("/")[-1] + " due to having missing values: ",
          f.shape[0] - f.dropna().shape[0])
    f.dropna(inplace=True)
    l1_texts = f['l1_text'].values.tolist()
    l2_texts = f['l2_text'].values.tolist()
    print(len(l1_texts), len(l2_texts))
    print(l1_texts[500])
    print("\n")
    print(l2_texts[500])
    return l1_texts, l2_texts


# Overriding the Dataset class required for the use of PyTorch's data loader classes.
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, l1_encodings, l2_encodings):
        self.l1_encodings = l1_encodings
        self.l2_encodings = l2_encodings

    def __getitem__(self, idx):
        item = {('l1_' + key): torch.tensor(val[idx]) for key, val in self.l1_encodings.items()}
        item2 = {('l2_' + key): torch.tensor(val[idx]) for key, val in self.l2_encodings.items()}
        item.update(item2)
        # item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.l1_encodings['attention_mask'])


class MyModel(nn.Module):
    # Each component other than the Transformer, are in a sequential layer (it is not required obviously, but it is
    # possible to stack them with other layers if desired)
    def __init__(self, base_model, n_classes, dropout=0.05):
        super().__init__()
        # self.base_model = base_model.to(CUDA_0)
        self.transformation_learner = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(BERT_EMB, BERT_EMB),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(BERT_EMB, BERT_EMB),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(BERT_EMB, BERT_EMB),
            nn.LeakyReLU()
        ).to(CUDA_0)

    def forward(self, input, **kwargs):
        l1_pooler_output = input
        # l2 = input2
        # if 'l1_attention_mask' in kwargs:
        #     l1_attention_mask = kwargs['l1_attention_mask']
            # l2_attention_mask = kwargs['l2_attention_mask']
        # else:
        #     print("my err: attention mask is not set, error maybe")
        # here we use only the CLS token
        # l1_pooler_output = self.base_model(l1.to(CUDA_0), attention_mask=l1_attention_mask.to(CUDA_0)).pooler_output
        myoutput = self.transformation_learner(l1_pooler_output)
        return myoutput


# The function to compute and print the performance measure scores using sklearn implementations.
def evaluate_model(labels, predictions, titlestr, logfile):
    myprint(titlestr, logfile)
    conf_matrix = confusion_matrix(labels, predictions)
    myprint("Confusion matrix- \n" + str(conf_matrix), logfile)
    acc_score = accuracy_score(labels, predictions)
    myprint('  Accuracy Score: {0:.2f}'.format(acc_score), logfile)
    myprint('Report', logfile)
    cls_rep = classification_report(labels, predictions)
    myprint(cls_rep, logfile)
    return f1_score(labels, predictions)  # return f-1 for positive class (sarcasm) as the early stopping measure.