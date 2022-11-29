"""
Running Prototypical network on our data-set.
Train:
For each train task-  compute a prototype for each class, which is the mean of all embeddings of support images from
this class. Then, it predicts the labels of the query set based on the information from the support set;
then it compares the predicted labels to ground truth query labels, and this gives the loss value.

Test: On query sets of test tasks- each sample with it's support.

Iterates over 30 seeds and return the averaged scores.

Source: https://github.com/JosephBless/Deep-Learning-Colab/blob/75c1137e919ac8f7b96acc63fd27f1d0a07f289b/Meta_Learning_few_shot_classifier.ipynb
"""

import os
import sys

from statistics import mean
from datetime import datetime
from datasets import Dataset

from sklearn.metrics import classification_report, confusion_matrix, \
    f1_score, accuracy_score, precision_score, recall_score

import numpy as np
import pandas as pd
import torch

from torch import nn, cuda, relu, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define csv files paths
root_path = "../"
train_path = root_path + "train_df.csv"
test_path = root_path + "test_df.csv"

# Define epochs
TRAIN_EPOCHS = 1
FT_EPOCHS = 50

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
f = open(f"../outputs/output_prototypical_{file_suffix}.txt", 'w')

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
    return torch.FloatTensor(x1_f).to(device)


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


class embedding_net(nn.Module):
    """
    Create an embedding of size 128.
    The layers are identical to the FSFM model except of the last layer used for the classification task.
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(INPUT_SIZE, INPUT_SIZE)
        self.do1 = nn.Dropout(0.1)
        self.layer2 = nn.Linear(INPUT_SIZE, 128)
        self.do2 = nn.Dropout(0.25)
        self.layer3 = nn.Linear(128, 128)

    def forward(self, x):
        x = relu(self.layer1(x))
        x = self.do1(x)
        x = relu(self.layer2(x))
        x = self.do2(x)
        x = relu(self.layer3(x))
        return x


class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
            self,
            support_images: torch.Tensor,
            support_labels: torch.Tensor,
            query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


def evaluate_on_one_task(
        model_f,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
) -> [int, int]:
    """
    Returns the predictions of query samples.
    """

    query_preds = \
        torch.max(model_f(support_images, support_labels, query_images).detach().data, dim=1)[1]
    all_labels.extend(query_labels.tolist())
    all_preds.extend(query_preds.tolist())

    return query_preds.tolist()


def train_step(
        model_f,
        ep: int,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
):
    optimizer.zero_grad()
    classification_scores = model_f(
        support_images.to(device), support_labels.to(device), query_images.to(device)
    )

    preds = np.argmax((classification_scores.cpu()).detach().numpy(), axis=1)
    labels = query_labels.cpu().numpy()
    acc = (preds == labels).astype(np.float32).mean().item()

    if ep == FT_EPOCHS - 1:
        try:
            print(confusion_matrix(labels, preds))
            print(classification_report(labels, preds, target_names=['0', '1']))

        except:
            print(confusion_matrix(labels, preds))

    loss = criterion(classification_scores, query_labels.to(device))
    loss.backward()
    optimizer.step()

    return acc, loss.item(), model_f


def init_loss_and_opt(model_w, weighted=True):
    """
     Creating the loss function (Cross Entropy) and optimizer (Adam)
    :param model_w: the training model
    :param weighted: use weighted loss or not
    :return: loss function and optimizer
    """
    # Weighted Loss
    if weighted:
        nSamples = [amount_true, amount_false]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        normedWeights = torch.FloatTensor(normedWeights).to(device)
        loss_function = nn.CrossEntropyLoss(weight=normedWeights).to(device)
    else:
        loss_function = CrossEntropyLoss()

    # epsilon definition for being the same as in Keras default values
    optimizer = optim.Adam(model_w.parameters(),eps=1e-07)

    return loss_function, optimizer


def train_process(model):
    """
    Train prototypical model on train tasks
    :param model: prototypical model

    :return: trained model
    """

    print(
        "############################################################################################################\n"
        "############################################## TRAINING TASKS ##############################################\n"
        "############################################################################################################")

    model.train()
    all_loss = []
    all_accs = []

    # For each train task:
    tr_df = train_df.groupby('task')
    for epoch in range(TRAIN_EPOCHS):
        print(f"TRAIN EPOCH: {epoch}")
        for name, group in tr_df:
            print(f"******************************************* {name} *******************************************")

            # Define support and query sets
            support_tr_set = Sample(group[group['set'] == 'support'])
            query_tr_set = Sample(group[group['set'] == 'query'])

            s_tr_params = {'batch_size': len(support_tr_set),
                           'shuffle': True,
                           }
            q_tr_params = {'batch_size': len(query_tr_set),
                           'shuffle': True,
                           }

            s_train_loader = DataLoader(support_tr_set, **s_tr_params)
            q_train_loader = DataLoader(query_tr_set, **q_tr_params)

            for _, data in enumerate(s_train_loader):
                features = [d for d in data['features']]
                support_vecs = torch.stack(features).to(device)
                support_labels = torch.tensor([d for d in data['label']]).to(device)

            for _, data in enumerate(q_train_loader):
                features = [d for d in data['features']]
                query_vecs = torch.stack(features).to(device)
                query_labels = torch.tensor([d for d in data['label']]).to(device)

            # Fine-tune on train task
            for j in range(FT_EPOCHS):
                acc, loss_value, model = train_step(model,epoch, support_vecs, support_labels, query_vecs,
                                      query_labels)
                print(f"{j}: ACC: {acc}\tLOSS: {loss_value}")
                if j == FT_EPOCHS - 1:
                    all_loss.append(loss_value)
                    all_accs.append(acc)

        try:
            print(f"\nMEAN ACC: {mean(all_accs)}\t ,MEAN LOSS: {mean(all_loss)}")
        except:
            print("NO training")
    return model


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


def eval_process(model):
    """
    For each test task- create a prototype with the support set and evaluate on the query set
    :param model: trained model
    :return:
    """
    print(
        "##########################################################################################################\n "
        "############################################## TESTING TASKS #############################################\n"
        "##########################################################################################################")

    test_labels = []
    test_preds = []

    # For each test task:
    ts_df = test_df.groupby('task')
    for name, group in ts_df:
        print(f"******************************************* {name} *******************************************")

        # Define support and query sets
        support_ts_set = Sample(group[group['set'] == 'support'])
        query_ts_set = Sample(group[group['set'] == 'query'])

        s_ts_params = {'batch_size': len(support_ts_set),
                       'shuffle': True,
                       'num_workers': 0
                       }
        q_ts_params = {'batch_size': len(query_ts_set),
                       'shuffle': True,
                       'num_workers': 0
                       }
        s_test_loader = DataLoader(support_ts_set, **s_ts_params, )
        q_test_loader = DataLoader(query_ts_set, **q_ts_params, )

        # Create support set
        for _, data in enumerate(s_test_loader):
            features = [d for d in data['features']]
            support_vecs = torch.stack(features).to(device)
            support_labels = torch.tensor([d for d in data['label']]).to(device)

        # Create query set
        for _, data in enumerate(q_test_loader):
            features = [d for d in data['features']]
            query_vecs = torch.stack(features).to(device)
            query_labels = torch.tensor([d for d in data['label']]).to(device)

        # Evaluate on query set
        print("\nEvaluation:")
        model.eval()
        with torch.no_grad():
            qr_preds = evaluate_on_one_task(model,
                                            support_vecs, support_labels, query_vecs, query_labels
                                            )
            print(
                f"Accuracy: {accuracy_score(query_labels.tolist(), qr_preds)}\n"
                f"F1-score:{f1_score(query_labels.tolist(), qr_preds)}\n")
            print(confusion_matrix(query_labels.tolist(), qr_preds))
            test_labels.extend(query_labels.tolist())
            test_preds.extend(qr_preds)
            try:
                print(classification_report(query_labels.tolist(), qr_preds, target_names=['0', '1']))
            except:
                ""

    print_scores(test_labels, test_preds)


def main():
    # Define prototypical model
    embedding_network = embedding_net().to(device)
    print(embedding_network)
    model = PrototypicalNetworks(embedding_network).to(device)
    global criterion, optimizer
    criterion, optimizer = init_loss_and_opt(model,weighted=True)

    trained_model = train_process(model)
    eval_process(trained_model)


if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    times = []
    for i in range(0, 30):
        print(i)
        SEED = i

        # Define seed for all libraries
        os.environ['PYTHONHASHSEED'] = str(SEED)
        torch.backends.cudnn.enabled = True
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        start = datetime.now()
        main()

        time = datetime.now() - start
        print("TIME:", time)
        times.append(time)

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
