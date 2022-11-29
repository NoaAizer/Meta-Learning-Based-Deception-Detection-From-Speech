"""
MAML network with FSFM model as the meta-learner.

- Train: -
For each training task:
* Fine-tunes a copy of the primary model (meta-learner)
* The weights of the copy are updated using the loss from the query samples in the task by stochastic gradient descent.
* At the end of each training epoch, the losses and gradients from all queries are accumulated.
* Calculates the derivative of the mean loss concerning the primary model’s weights, and updates those weights.

- Test: -
For each testing task:
* Create a copy of the primary model is fine-tuned on the support set.
* Evaluates on the query set.


Sources:
https://towardsdatascience.com/advances-in-few-shot-learning-reproducing-results-in-pytorch-aba70dee541d
https://github.com/oscarknagg/few-shot/blob/master/few_shot/maml.py
"""
import gc
import os
import random
import sys

from collections import OrderedDict
from statistics import fmean, mean
from typing import Callable
from datetime import datetime
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score, recall_score, \
    precision_score

import numpy as np
import pandas as pd

import torch
from torch import nn, optim, cuda
from torch.nn import functional as F
from torch.nn import Module
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Optimizer

import warnings

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Define csv files paths
root_path = "../"
train_path = root_path + "train_df.csv"
test_path = root_path + "test_df.csv"

# Define epochs
TRAIN_EPOCHS = 50
INNER_TRAIN_EPOCHS = 15# 2
LR = 0.001

# If Wav2Vec 2.0 embedding, otherwise five-sound-features
WAV2VEC = False

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

train_df = pd.read_csv(train_path, delimiter=',')
amount_false = sum(train_df['label'])
amount_true = len(train_df) - amount_false
test_df = pd.read_csv(test_path, delimiter=',')

if WAV2VEC:
    INPUT_SIZE = 1024
    vecs_dir = "../wav2vec_vecs"
    file_suffix = "wav2vec"
else:
    INPUT_SIZE = 193
    vecs_dir = "../fsfm_dir"
    file_suffix = "five_fe"

# Write results to file
f = open(f"../outputs/output_MAML_{file_suffix}.txt", 'w')
sys.stdout = f

all_labels = []
all_preds = []

ALL_ACCS = []
ALL_PRECISIONS = []
ALL_RECALLS = []
ALL_F1S = []


def speech_file_to_array_fn(sample):
    """
    Convert each audio sample to its embedding vector.
    :param sample: audio sample
    :return: a tensor contains the embedding
    """
    x1_f = np.load(f'{vecs_dir}/{sample["file_name"]}.npy', allow_pickle=True)
    return torch.Tensor(x1_f).to(device)


class Sample(Dataset):
    """
    A class represents each audio sample.

    features - the embedding tensor.\n
    label - 0 = True, 1 = False.\n
    path - the path to the audio sample.\n
    task - the name of the task to audio sample belongs to.\n
    set - support/query.
    """

    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.df = dataframe

    def __getitem__(self, index):
        return {
            'features': speech_file_to_array_fn(self.df.iloc[index]).to(device),
            'label': self.df.iloc[index]['label'],
            'path': self.df.iloc[index]['path'],
            'task': self.df.iloc[index]['task'],
            'set': self.df.iloc[index]['set']
        }

    def __len__(self):
        return self.len


class FSFM(nn.Module):
    """
    Use FSFM model as the meta-learner
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(INPUT_SIZE, INPUT_SIZE)
        self.do1 = nn.Dropout(0.1)
        self.layer2 = nn.Linear(INPUT_SIZE, 128)
        self.do2 = nn.Dropout(0.25)
        self.layer3 = nn.Linear(128, 128)
        self.do3 = nn.Dropout(0.5)
        self.layer4 = nn.Linear(128, 2)


    def replace_grad(parameter_gradients, parameter_name):
        """Creates a backward hook function that replaces the calculated gradient
        with a precomputed value when .backward() is called.

        See
        https://pytorch.org/docs/stable/autograd.html?highlight=hook#torch.Tensor.register_hook
        for more info
        """

        def replace_grad_(module):
            return parameter_gradients[parameter_name]

        return replace_grad_

    def functional_forward(self, x: torch.Tensor, weights: dict):
        """Performs a forward pass of the network using the PyTorch functional API."""
        x = F.relu(F.linear(x, weights['layer1.weight'], weights['layer1.bias']))
        x = self.do1(x)
        x = F.relu(F.linear(x, weights['layer2.weight'], weights['layer2.bias']))
        x = self.do2(x)
        x = F.relu(F.linear(x, weights['layer3.weight'], weights['layer3.bias']))
        x = self.do3(x)
        x = F.linear(x, weights['layer4.weight'], weights['layer4.bias'])
        return x


def fine_tune_test_task(model, loss_fn,support_vecs, support_labels):
    """
    Fine tune the copy model on the support set of the current test taskץ
    :param model: The primary trained model
    :param loss_fn: Loss function to calculate between predictions and outputs
    :param support_vecs: Support set embedding vectors
    :param support_labels: Support set labels
    :return: weights of fine-tuned model
    """
    # copy_model.train()
    print("----------------------------- TRAIN -----------------------------")
    # Update the copy model of current task
    # Create a fast (copy) model using the current meta model weights
    fast_weights = OrderedDict(model.named_parameters())

    for inner_batch in range(INNER_TRAIN_EPOCHS):
        # Perform update of model weights
        logits = model.functional_forward(support_vecs, fast_weights)
        loss = loss_fn(logits, support_labels)
        gradients = torch.autograd.grad(loss, fast_weights.values(), )

        # Update weights manually
        fast_weights = OrderedDict(
            (name, param - LR * grad)
            for ((name, param), grad) in zip(fast_weights.items(), gradients)
        )
        print(loss)

    return fast_weights


def print_scores(test_labels, test_preds):
    """
    Print all current scores, for current SEED and the mean of all previous ones.
    :param test_labels: labels of all query sets of test tasks
    :param test_preds: predicted labels of all query sets of test tasks
    :return:
    """
    # Print all tasks scores
    accuracy_all_tasks = accuracy_score(test_labels, test_preds)
    precision_all_tasks = precision_score(test_labels, test_preds)
    recall_all_tasks = recall_score(test_labels, test_preds)
    f1_all_tasks = f1_score(test_labels, test_preds)

    print(f" ############################## SEED {SEED} SCORES: ###############################")

    print(confusion_matrix(test_labels, test_preds))
    print(f"Mean Accuracy: {accuracy_all_tasks}")
    print(f"Mean Precision: {precision_all_tasks}")
    print(f"Mean Recall: {recall_all_tasks}")
    print(f"Mean F1-score: {f1_all_tasks}")

    ALL_ACCS.append(accuracy_all_tasks)
    ALL_PRECISIONS.append(precision_all_tasks)
    ALL_RECALLS.append(recall_all_tasks)
    ALL_F1S.append(f1_all_tasks)

    print(f" ##############################  MEAN SCORES OF ALL CURRENT SEEDS: ###############################")

    print(confusion_matrix(all_labels, all_preds))
    print(f"Current Mean Accuracy: {mean(ALL_ACCS)}")
    print(f"Current Mean Precision: {mean(ALL_PRECISIONS)}")
    print(f"Current Mean Recall: {mean(ALL_RECALLS)}")
    print(f"Current Mean F1-score: {mean(ALL_F1S)}")


def eval_testing_tasks(model,loss_fn):
    """
    Evaluate MAML model on testing tasks.\n
    For each testing task:\n
    * Create a copy of the primary model is fine-tuned on the support set.\n
    * Evaluates on the query set.\n
    :param model: The primary trained model
    :param loss_fn: Loss function to calculate between predictions and outputs
    :return:
    """

    print(
        "##########################################################################################################\n "
        "############################################## TESTING TASKS #############################################\n"
        "##########################################################################################################")

    test_preds = []
    test_labels = []

    # For each testing task:
    ts_df = test_df.groupby('task')
    for name, group in ts_df:
        print(f"*************************************** {name} ***************************************")

        # Define support and query sets
        support_ts_set = Sample(group[group['set'] == 'support'])
        query_ts_set = Sample(group[group['set'] == 'query'])

        s_ts_params = {'batch_size': len(support_ts_set),
                       'shuffle': True,

                       }
        q_ts_params = {'batch_size': len(query_ts_set),
                       'shuffle': False,
                       }

        s_test_loader = DataLoader(support_ts_set, **s_ts_params)
        q_test_loader = DataLoader(query_ts_set, **q_ts_params)

        for _, data in enumerate(s_test_loader):
            features = [d for d in data['features']]
            support_vecs = torch.stack(features).to(device)
            support_labels = torch.tensor([d for d in data['label']]).to(device)

        for _, data in enumerate(q_test_loader):
            features = [d for d in data['features']]
            query_vecs = torch.stack(features).to(device)
            query_labels = torch.tensor([d for d in data['label']]).to(device)

        # For each query set, fine-tune on support set for "inner_train_steps".
        ft_weights = fine_tune_test_task(model=model,loss_fn=loss_fn,
                                         support_vecs=support_vecs, support_labels=support_labels)

        print("----------------------------- VAL -----------------------------")
        # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
        # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
        model.eval()
        with torch.no_grad():
            # Do a pass of the model on the validation data from the current task
            logits = model.functional_forward(query_vecs, ft_weights)

            # Get post-update accuracies and f1-score
            y_pred = logits.softmax(dim=1)
            preds = np.argmax((y_pred.cpu()).detach().numpy(), axis=1)
            labels = query_labels.cpu().numpy()
            acc = (preds == labels).astype(np.float32).mean().item()

            print(
                "Evaluation:\n"
                f"Accuracy: {acc}\n"
                f"F1-score:{f1_score(labels, preds, pos_label=1, average=None)}\n")
            print(confusion_matrix(labels, preds))
            test_labels.extend(labels)
            test_preds.extend(preds)

            # Print evaluation results
            try:
                print(classification_report(labels, preds, target_names=['0', '1']))
            except:
                ""
    print_scores(test_labels, test_preds)


def meta_gradient_step(model: Module,
                       optimiser: Optimizer,
                       loss_fn: Callable,
                       order: int,
                       epochs: int,
                       inner_train_steps: int,
                       train: bool):
    """
    Perform a gradient step on a meta-learner.
    # Arguments
        model: Base model of the meta-learner being trained
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        order: Whether to use 1st order MAML (update meta-learner weights with gradients of the updated weights on the
            query set) or 2nd order MAML (use 2nd order updates by differentiating through the gradients of the updated
            weights on the query with respect to the original weights).
        epochs: Number of iteration iterating over all training tasks.
        inner_train_steps: Number of gradient steps to fit the fast weights during each inner update
        train: Whether to update the meta-learner weights at the end of the episode.

    """

    create_graph = (True if order == 2 else False) and train

    tr_df = train_df.groupby('task')

    # Training the primary model
    for i in range(epochs):

        print('EPOCH: ', i)

        task_gradients = []
        task_losses = []
        task_predictions = []
        accs = []
        f_s = []

        # For each training task:
        for name, group in tr_df:
            print(f"******************************************* {name} *******************************************")

            # Create a fast (copy) model using the current meta model weights
            fast_weights = OrderedDict(model.named_parameters())

            # Define support and query sets
            support_tr_set = Sample(group[group['set'] == 'support'])
            query_tr_set = Sample(group[group['set'] == 'query'])

            s_tr_params = {'batch_size': len(support_tr_set),
                           'shuffle': True,
                           }
            q_tr_params = {'batch_size': len(query_tr_set),
                           'shuffle': False,
                           }
            s_train_loader = DataLoader(support_tr_set, **s_tr_params)

            # Weighted Sampler for query set
            y = group[group['set'] == 'query']['label']
            counts = np.bincount(y)
            labels_weights = 1. / counts
            weights = labels_weights[y]
            q_train_loader = DataLoader(query_tr_set, **q_tr_params,
                                        sampler=WeightedRandomSampler(weights, len(weights)))



            for _, data in enumerate(s_train_loader):
                features = [d for d in data['features']]
                support_vecs = torch.stack(features).to(device)
                support_labels = torch.tensor([d for d in data['label']]).to(device)

            # For each batch of query - fine tune
            for _, data in enumerate(q_train_loader):
                features = [d for d in data['features']]
                query_vecs = torch.stack(features).to(device)
                query_labels = torch.tensor([d for d in data['label']]).to(device)

            # Train the model for `inner_train_steps` iterations
            for inner_batch in range(inner_train_steps):
                # print(f"------------------------- inner train epoch {inner_batch+1}:-------------------------")

                # Perform update of model weights
                logits = model.functional_forward(support_vecs, fast_weights)
                loss = loss_fn(logits, support_labels)
                gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

                preds = np.argmax((logits.cpu()).detach().numpy(), axis=1)
                labels = support_labels.cpu().numpy()
                acc = (preds == labels).astype(np.float32).mean().item()

                # Update weights manually
                fast_weights = OrderedDict(
                    (name, param - LR * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )

                # print("----------------------------- VAL -----------------------------")

            # Do a pass of the model on the validation data from the current task
            logits = model.functional_forward(query_vecs, fast_weights)
            loss = loss_fn(logits, query_labels)
            loss.backward(retain_graph=True)

            # Get post-update accuracies
            y_pred = logits.softmax(dim=1)
            task_predictions.append(y_pred)

            preds = np.argmax((y_pred.cpu()).detach().numpy(), axis=1)
            labels = query_labels.cpu().numpy()

            acc = (preds == labels).astype(np.float32).mean().item()
            accs.append(acc)

            f1 = f1_score(labels, preds, average=None)
            f_s.append(f1[-1])

            # Print classification report for current task query set
            try:
                print("ACC:", acc)
                print("LOSS:", loss.item())
                print(classification_report(labels, preds, target_names=['0', '1']))
            except:
                ""

            # Accumulate losses and gradients
            task_losses.append(loss)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}
            task_gradients.append(named_grads)

        if order == 1:
            if train:
                sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
                                      for k in task_gradients[0].keys()}
                hooks = []
                for name, param in model.named_parameters():
                    hooks.append(
                        param.register_hook(model.replace_grad(sum_task_gradients, name))
                    )

                model.train()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                for h in hooks:
                    h.remove()

            loss, pred = torch.stack(task_losses).mean(), torch.cat(task_predictions)

        elif order == 2:
            # Update the weights of the primary model
            model.train()
            optimiser.zero_grad()
            meta_batch_loss = torch.stack(task_losses).mean()
            if train:
                meta_batch_loss.backward()
                optimiser.step()

            loss, pred = meta_batch_loss, torch.cat(task_predictions)
            avg_acc = fmean(accs)
            avg_f = fmean(f_s)
            print(f"TRAIN QUERIES EPOCH {i}: \nLOSS:{loss.item()} \nACC: {avg_acc} \nF1: {avg_f}\n")

    # Evaluate on testing task only after the last train epoch
    eval_testing_tasks(model=model, loss_fn=loss_fn)


def main():
    # Define the primary model
    model = FSFM().to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # epsilon definition for being the same as in Keras default values
    optimizer = optim.Adam(model.parameters(),eps=1e-07)

    # Execution
    meta_gradient_step(model=model, optimiser=optimizer, loss_fn=criterion,
                       order=2, epochs=TRAIN_EPOCHS, inner_train_steps=INNER_TRAIN_EPOCHS,train=True)


if __name__ == '__main__':

    times = []
    for i in range(10, 30):
        print(i)
        SEED = i

        # Define seed for all libraries
        os.environ['PYTHONHASHSEED'] = str(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        torch.cuda.empty_cache()

        start = datetime.now()
        main()

        time = datetime.now() - start
        print("TIME:", time)
        times.append(time)

        gc.collect()

        # Calculate the current averaged time of all seeds' executions
        avgTime = pd.to_datetime(pd.Series(times).values.astype('datetime64[D]')).mean()
        print(f"Current Mean Time:{avgTime}")

    print("***************************** Final Scores *****************************")

    print(ALL_ACCS)
    print(ALL_F1S)
    print(f"Mean Accuracy:{mean(ALL_ACCS)}")
    print(f"Mean Precisions:{mean(ALL_PRECISIONS)}")
    print(f"Mean Recall:{mean(ALL_RECALLS)}")
    print(f"Mean F1s:{mean(ALL_F1S)}")

    # Calculate the average time of all seeds' executions
    avgTime = pd.to_datetime(pd.Series(times).values.astype('datetime64[D]')).mean()
    print(f"Mean Time:{avgTime}")
    f.close()
