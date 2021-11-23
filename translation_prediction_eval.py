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
from myutils import myprint, load_data, MyDataset, evaluate_model
from myutils import MyModel

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


TEST_PATH = "/s/bach/h/proj/COVID-19/smps/machine_translation/mt_pcp/mbert_transformation_learner/test.tsv"
LOAD_PATH = "/s/lovelace/c/nobackup/iray/sinamps/tempmodels/" \
            "mbert_transformation_learner_(basedOnUtils)-EarlyStoppedFinal-time:1637639904.5309534"

# Configuration variables to choose the pre-trained model you want to use and other training settings:
# the pre-trained model name from huggingface transformers library names:
PRE_TRAINED_MODEL = 'bert-base-multilingual-cased'
# it can be from the followings for example: 'digitalepidemiologylab/covid-twitter-bert-v2',
#                                            'bert-large-uncased',
#                                            'vinai/bertweet-base'
#                                            'xlnet-base-cased'

MAXTOKENS = 512
NUM_EPOCHS = 400  # default maximum number of epochs
BERT_EMB = 768  # set to either 768 or 1024 for BERT-Base and BERT-Large models respectively
BS = 4  # batch size
INITIAL_LR = 1e-3  # initial learning rate
save_epochs = [1, 2, 3, 4, 5, 6, 7]  # these are the epoch numbers (starting from 1) to test the model on the test set
# and save the model checkpoint.
EARLY_STOP_PATIENCE = 10  # If model does not improve for this number of epochs, training stops.

# Setting GPU cards to use for training the model. Make sure you read our paper to figure out if you have enough GPU
# memory. If not, you can change all of them to 'cpu' to use CPU instead of GPU. By the way, two 24 GB GPU cards are
# enough for current configuration, but in case of developing based on this you may need more (that's why there are
# three cards declared here)
CUDA_0 = 'cuda:1'
CUDA_1 = 'cuda:1'
CUDA_2 = 'cuda:1'


def l2dist(a, b):
    return sum(((a - b) ** 2))


# The function to do a forward pass of the network.
def feed_model(base_model, model, data_loader, dataset):
    outputs_all = []
    l2_pooler_output_all = []
    other_l2_pooler_outputs = []
    c = 0
    for index in range(len(dataset)):
        row = dataset[index]
        # print(index, row['l2_input_ids'][:20])
        # transformer tokenizer
        l1_pooler_output = base_model(torch.unsqueeze(row['l1_input_ids'].to(CUDA_0), 0),
                                      attention_mask=torch.unsqueeze(row['l1_attention_mask'].to(CUDA_0), 0),
                                      return_dict=True).last_hidden_state[:, 0, :]
        mapped_output = model(l1_pooler_output)
        l2_pooler_output = base_model(torch.unsqueeze(row['l2_input_ids'].to(CUDA_0), 0),
                                      attention_mask=torch.unsqueeze(row['l2_attention_mask'].to(CUDA_0), 0),
                                      return_dict=True).last_hidden_state[:, 0, :]
        rand_index = random.sample([i for i in range(0, len(dataset)) if i not in [index]], 4)
        for j in rand_index:
            outj = base_model(torch.unsqueeze(dataset[j]['l2_input_ids'].to(CUDA_0), 0),
                       attention_mask=torch.unsqueeze(dataset[j]['l2_attention_mask'].to(CUDA_0), 0),
                       return_dict=True).last_hidden_state[:, 0, :]
            outj = torch.squeeze(outj).detach().cpu().numpy()
            other_l2_pooler_outputs.append(outj)
        l2_pooler_output = torch.squeeze(l2_pooler_output).detach().cpu().numpy()
        other_l2_pooler_outputs.append(l2_pooler_output)
        mapped_output = mapped_output.detach().cpu().numpy()
        dists = []
        for x in other_l2_pooler_outputs:
            dists.append(np.linalg.norm(x - mapped_output))
        m = np.argmin(dists)
        if m == 4:
            outputs_all.append(1)
        else:
            outputs_all.append(0)
        other_l2_pooler_outputs.clear()
        dists.clear()
        # outputs = mapped_output.detach().cpu().numpy()
        # l2_pooler_output = l2_pooler_output.detach().to('cpu').numpy()
        # outputs_all.extend(outputs)
        # l2_pooler_output_all.extend(l2_pooler_output)
        # del outputs, l2_pooler_output
    return outputs_all


if __name__ == '__main__':
    args = sys.argv
    epochs = NUM_EPOCHS
    logfile = open('log_file_' + args[0].split('/')[-1][:-3] + str(time.time()) + '.txt', 'w')
    myprint("Please wait for the model to download and load sub-models, getting a few warnings is OK.", logfile)
    test_l1_texts, test_l2_texts = load_data(TEST_PATH)
    # print(test_l1_texts[:10])
    # print(test_l2_texts[:10])
    # exit(0)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
    tokenizer.model_max_length = MAXTOKENS
    test_l1_encodings = tokenizer(test_l1_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    test_l2_encodings = tokenizer(test_l2_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    # print(test_l2_encodings['input_ids'][:10])
    test_dataset = MyDataset(test_l1_encodings, test_l2_encodings)
    test_data_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)  # shuffle False for reproducibility
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL).to(CUDA_0)
    model = MyModel(base_model=base_model, n_classes=2)
    # If you want to load an already fine-tuned model and continue its training, uncomment the next line.
    model.load_state_dict(torch.load(LOAD_PATH))
    model.eval()  # take model to evaluation mode
    base_model.eval()

    total_steps = len(test_data_loader) * epochs
    test_outputs = feed_model(base_model, model, test_data_loader, test_dataset)
    # labels = [1 for i in range(len(test_outputs))]
    trues = np.sum(test_outputs)
    all = len(test_outputs)
    acc = trues/all * 100
    print("Accuracy = ", acc, " %")
    # evaluate_model(labels, test_outputs, 'Test set Result', logfile)
    exit(0)
    # Training loop:
    for epoch in range(epochs):
        print(' EPOCH {:} / {:}'.format(epoch+1, epochs))
        outputs_all = []
        l2_pooler_output_all = []
        for step, batch in enumerate(train_data_loader):
            if step % 100 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_data_loader)))
            optim.zero_grad()
            # l1_input_ids = batch['l1_input_ids']  # 'input_ids' are the index of tokens in model's dictionary
            # l1_attention_mask = batch['l1_attention_mask']  # 'attention_mask' indicates the non-padding tokens
            l1_pooler_output = base_model(batch['l1_input_ids'].to(CUDA_0),
                                          attention_mask=batch['l1_attention_mask'].to(CUDA_0),
                                          return_dict=True).last_hidden_state[:, 0, :]
            outputs = model(l1_pooler_output)
            l2_pooler_output = base_model(batch['l2_input_ids'].to(CUDA_0),
                                          attention_mask=batch['l2_attention_mask'].to(CUDA_0),
                                          return_dict=True).last_hidden_state[:, 0, :]
            loss = loss_model(outputs, l2_pooler_output)
            loss.backward()
            optim.step()
            scheduler.step()
            # current_loss += loss.item()
            outputs = outputs.detach().cpu().numpy()
            l2_pooler_output = l2_pooler_output.detach().to('cpu').numpy()
            outputs_all.extend(outputs)
            l2_pooler_output_all.extend(l2_pooler_output)
            del outputs, l2_pooler_output
        evaluate_model(l2_pooler_output_all, outputs_all, 'Train set Result epoch ' + str(epoch+1), logfile)
        del l2_pooler_output_all, outputs_all
        model.eval()
        val_l2s, val_outputs = feed_model(base_model, model, val_data_loader)
        val_err = evaluate_model(val_l2s, val_outputs, 'Validation set Result epoch ' + str(epoch+1), logfile)
        del val_l2s, val_outputs
        myprint("------------------------------- Val Error at epoch " + str(epoch+1) + " : " + str(val_err), logfile)
        # The early stopping logic:
        if best_val_err is None:
            best_val_err = val_err
            torch.save(model.state_dict(), (SAVE_PATH + args[0].split('/')[-1][:-3] + '_checkpoint'))
        elif val_err <= best_val_err:
            best_val_err = val_err
            myprint("Better Error; saving Model", logfile)
            patience_counter = 0
            torch.save(model.state_dict(), (SAVE_PATH + args[0].split('/')[-1][:-3] + '_checkpoint'))
        else:
            patience_counter = patience_counter + 1
            myprint("Worse Error; Patience Counter:" + str(patience_counter), logfile)
            if patience_counter >= EARLY_STOP_PATIENCE:  # patience reached, stop training
                # Load the last best model:
                model.load_state_dict(torch.load(SAVE_PATH + args[0].split('/')[-1][:-3] + '_checkpoint'))
                # Test the model on the testing set:
                test_l2s, test_outputs = feed_model(base_model, model, test_data_loader)
                evaluate_model(test_l2s, test_outputs, 'Test set Result epoch ' + str(epoch + 1), logfile)
                torch.save(model.state_dict(), (SAVE_PATH + args[0].split('/')[-1][:-3] + '-EarlyStoppedFinal-' +
                                                'time:' + str(time.time())))
                break
        model.train()
    del train_data_loader, val_data_loader, test_data_loader
    # End of main

