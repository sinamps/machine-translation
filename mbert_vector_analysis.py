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
from myutils import load_data, myprint, MyDataset
# from vlad.mynextvlad import NeXtVLAD  # this imports our implementation of the NeXtVLAD layer
# from vlad.internetnextvlad import NextVLAD  # this imports an implementation of the NeXtVLAD layer taken from
# https://www.kaggle.com/gibalegg/mtcnn-nextvlad#NextVLAD

# Setting manual seed for various libs for reproducibility purposes.
torch.manual_seed(7)
random.seed(7)
np.random.seed(7)
# Setting PyTorch's required configuration variables for reproducibility.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
# To run the code in a reproducibale way, use the following running parameter for CUDA 10.2 or higher:
# CUBLAS_WORKSPACE_CONFIG=:16:8 python tbinv_earlystop.py
# If you do not care about reproducibility, you can comment above configs and run the script without the parameter

# Path to data. Make sure you edit them properly to point to your local
# datasets.
TRAIN_PATH = "/s/bach/h/proj/COVID-19/smps/machine_translation/mt_pcp/parsinlu/all_nonrel.tsv"
VAL_PATH = "/s/bach/h/proj/COVID-19/smps/machine_translation/mt_pcp/parsinlu/all_nonrel_dev.tsv"
TEST_PATH = "/s/bach/h/proj/COVID-19/smps/machine_translation/mt_pcp/parsinlu/all_nonrel_test.tsv"
SAVE_PATH = "/s/lovelace/c/nobackup/iray/sinamps/tempmodels/"
# If you want to load an already fine-tuned model and continue its training, uncomment and edit the following line.
# LOAD_PATH = "/s/lovelace/c/nobackup/iray/sinamps/claim_project/2021-06-01_models/jun10_tbinv_ensemble_with_bigger_batch-final-epoch-15"

# Configuration variables to choose the pre-trained model you want to use and other training settings:
# the pre-trained model name from huggingface transformers library names:
PRE_TRAINED_MODEL = 'bert-base-multilingual-cased'
# it can be from the followings for example: 'digitalepidemiologylab/covid-twitter-bert-v2',
#                                            'bert-large-uncased',
#                                            'vinai/bertweet-base'
#                                            'xlnet-base-cased'

MAXTOKENS = 512
NUM_EPOCHS = 2000  # default maximum number of epochs
BERT_EMB = 768  # set to either 768 or 1024 for BERT-Base and BERT-Large models respectively
BS = 8  # batch size
INITIAL_LR = 1e-5  # initial learning rate
save_epochs = [1, 2, 3, 4, 5, 6, 7]  # these are the epoch numbers (starting from 1) to test the model on the test set
# and save the model checkpoint.
EARLY_STOP_PATIENCE = 30  # If model does not improve for this number of epochs, training stops.

# Setting GPU cards to use for training the model. Make sure you read our paper to figure out if you have enough GPU
# memory. If not, you can change all of them to 'cpu' to use CPU instead of GPU. By the way, two 24 GB GPU cards are
# enough for current configuration, but in case of developing based on this you may need more (that's why there are
# three cards declared here)
CUDA_0 = 'cuda:1'
CUDA_1 = 'cuda:1'
CUDA_2 = 'cuda:1'


# The function to compute and print the performance measure scores using sklearn implementations.
def evaluate_model(target, predictions, titlestr, logfile):
    myprint(titlestr, logfile)
    e = mean_squared_error(target, predictions)
    myprint("MSE Error- \n" + str(e), logfile)
    return e


# The function to do a forward pass of the network.
def feed_model(base_model, model, data_loader):
    outputs_all = []
    l2_pooler_output_all = []
    for batch in data_loader:
        # transformer tokenizer
        l1_pooler_output = base_model(batch['l1_input_ids'].to(CUDA_0),
                                      attention_mask=batch['l1_attention_mask'].to(CUDA_0),
                                      return_dict=True).last_hidden_state[:, 0, :]
        outputs = model(l1_pooler_output)
        l2_pooler_output = base_model(batch['l2_input_ids'].to(CUDA_0),
                                      attention_mask=batch['l2_attention_mask'].to(CUDA_0),
                                      return_dict=True).last_hidden_state[:, 0, :]
        outputs = outputs.detach().cpu().numpy()
        l2_pooler_output = l2_pooler_output.detach().to('cpu').numpy()
        outputs_all.extend(outputs)
        l2_pooler_output_all.extend(l2_pooler_output)
        del outputs, l2_pooler_output
    return l2_pooler_output_all, outputs_all


class MyModel(nn.Module):
    # Each component other than the Transformer, are in a sequential layer (it is not required obviously, but it is
    # possible to stack them with other layers if desired)
    def __init__(self, base_model, n_classes, dropout=0.05):
        super().__init__()
        # self.base_model = base_model.to(CUDA_0)
        self.transformation_learner = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(BERT_EMB, BERT_EMB),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(BERT_EMB, BERT_EMB),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(BERT_EMB, BERT_EMB),
            # nn.LeakyReLU()
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


if __name__ == '__main__':
    args = sys.argv
    epochs = NUM_EPOCHS
    # logfile = open('log_file_' + args[0].split('/')[-1][:-3] + str(time.time()) + '.txt', 'w')
    # myprint("Please wait for the model to download and load sub-models, getting a few warnings is OK.", logfile)
    # train_l1_texts, train_l2_texts = load_data(TRAIN_PATH)
    fa = ["کتاب", "کتاب"]
    en = ["book", "food"]
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
    tokenizer.model_max_length = MAXTOKENS
    l1_encodings = tokenizer(fa, truncation=True, padding='max_length', max_length=MAXTOKENS)
    l2_encodings = tokenizer(en, truncation=True, padding='max_length', max_length=MAXTOKENS)
    dataset = MyDataset(l1_encodings, l2_encodings)
    data_loader = DataLoader(dataset, batch_size=BS, shuffle=False)  # shuffle False for reproducibility
    base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL).to(CUDA_0)
    base_model.eval()
    cos_s = torch.nn.CosineSimilarity()
    for step, batch in enumerate(data_loader):
        l1_vector = base_model(batch['l1_input_ids'].to(CUDA_0),
                                      attention_mask=batch['l1_attention_mask'].to(CUDA_0),
                                      return_dict=True).last_hidden_state[:, 1, :]
        l2_vector = base_model(batch['l2_input_ids'].to(CUDA_0),
                                      attention_mask=batch['l2_attention_mask'].to(CUDA_0),
                                      return_dict=True).last_hidden_state[:, 1, :]
        print("Similarities between l1 and l2 words are ", cos_s(l1_vector, l2_vector))
    # End of main

