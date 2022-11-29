"""
Running CHAML model with FOUR-input samples.
Train: all train tasks (shuffled), each with four examples from the same task.
Test: On query sets of test tasks- each sample with it's support.

Iterates over 30 seeds and return the averaged scores.

* At the first time you run this program, use create_train_support_set.py to create the train_CHAML.csv file
"""
import gc
import os
import sys
import ast
import random

from datetime import datetime
from statistics import mean
from thinc.util import to_categorical

import numpy as np
import pandas as pd
import tensorflow as tf

import keras.backend
from keras import Input
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Dense, Dropout, concatenate, multiply

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, \
    f1_score
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder

# Use GPU and define setting for getting reproducible results
os.environ['TF_DETERMINISTIC_OPS'] = '1'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Define Batch-size and epochs
TRAIN_BATCH_SIZE = 512
EPOCHS = 50

# If Wav2Vec 2.0 embedding, otherwise five-sound-features
WAV2VEC = True

root_path = "../"
train_path = root_path + "train_CHAML.csv"
test_path = root_path + "test_df.csv"

all_labels = []
all_preds = []

ALL_ACCS = []
ALL_RECALLS = []
ALL_PRECISIONS = []
ALL_F1S = []

if WAV2VEC:
    INPUT_SIZE = 1024
    vecs_dir = "../wav2vec_vecs"
    saved_loaded_vecs_dir = "../wav2vec_dfs_vecs"
    file_suffix = "wav2vec"

else:
    INPUT_SIZE = 193
    vecs_dir = "../fsfm_dir"
    saved_loaded_vecs_dir = "../fsfm_dfs_vecs"
    file_suffix = "five_fe"

# Write results to file
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
f = open(f"../outputs/output_CHAML_{file_suffix}1.txt", 'w')
sys.stdout = f


def CHAML_model():
    # query sample
    query = Input(shape=(INPUT_SIZE,))
    # support samples
    t1 = Input(shape=(INPUT_SIZE,))
    t2 = Input(shape=(INPUT_SIZE,))
    f1 = Input(shape=(INPUT_SIZE,))
    f2 = Input(shape=(INPUT_SIZE,))

    # CHAML- core

    D1 = Dense(INPUT_SIZE * 3, activation='relu')
    DO1 = Dropout(0.2)
    D2 = Dense(INPUT_SIZE * 2, activation='relu')
    DO2 = Dropout(0.2)
    D3 = Dense(INPUT_SIZE * 2, activation='relu')

    # TRUE - preprocess
    T_mul = multiply([t1, t2])
    T_con = concatenate([t1, t2, T_mul])
    t_d1 = D1(T_con)
    t_do1 = DO1(t_d1)
    t_d2 = D2(t_do1)
    t_do2 = DO2(t_d2)
    T_embedding = D3(t_do2)

    # FALSE - preprocess
    F_mul = multiply([f1, f2])
    F_con = concatenate([f1, f2, F_mul])
    f_d1 = D1(F_con)
    f_do1 = DO1(f_d1)
    f_d2 = D2(f_do1)
    f_do2 = DO2(f_d2)
    F_embedding = D3(f_do2)

    # Classifier (FSFM)
    merge = concatenate([query, T_embedding, F_embedding])
    fc1 = Dense(INPUT_SIZE, activation='relu')(merge)
    do1 = Dropout(0.1)(fc1)
    fc2 = Dense(128, activation='relu', )(do1)
    do2 = Dropout(0.25)(fc2)
    fc3 = Dense(128, activation='relu', )(do2)
    do3 = Dropout(0.5)(fc3)
    output = Dense(2, activation='softmax', )(do3)

    model = Model(inputs=[query, t1, t2, f1, f2], outputs=output)
    print(model.summary())
    # plot graph
    plot_model(model, to_file='../outputs/CHAML_model.png', show_shapes=True, show_layer_names=True,
               show_layer_activations=True)

    # Specify the optimizer, and compile the model with loss functions for both outputs
    opt = tf.keras.optimizers.Adam()

    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    return model


def read_train_data(samples_df, num=0):
    """
    Loading the train features from the .npy file into the df.
    Save full query(x_1), support (x_4), and label (y_1) vectors each in a .npy file
    for each execution- only at the first execution time.
    :return:
    """
    df_train = pd.read_csv(root_path + "train_df.csv")
    try:
        os.mkdir(saved_loaded_vecs_dir)
    except:
        ""

    x_1 = []
    y_1 = []
    features = []

    # loading the features
    for row in samples_df.iterrows():
        # Load query embedding from npy file
        x1_f = np.load(f'{vecs_dir}/{row[1]["1_name"]}.npy', allow_pickle=True).reshape([1, INPUT_SIZE])
        x_1.append(np.concatenate(x1_f[:]))
        # Use label from train_df
        label = df_train[df_train["file_name"] == row[1]["1_name"]]["label"].item()
        y_1.append(label)

        # Load support set
        try:
            all_4 = ast.literal_eval(row[1]['4_name'])
        except:
            all_4 = row[1]['4_name']
        temp = []
        for name in all_4:
            x4_f = np.load(f'{vecs_dir}/{name}.npy', allow_pickle=True)
            temp.extend(x4_f)
        features.append(temp)

    samples_df["4_features"] = features
    samples_df["1_features"] = x_1
    samples_df["1_label"] = y_1
    # samples_df.to_csv(df_path, index=False)

    # Save to be used in next executions
    np.save(f'{saved_loaded_vecs_dir}/x_1_{num}_{SEED}.npy', np.array(x_1))
    np.save(f'{saved_loaded_vecs_dir}/y_1_{num}_{SEED}.npy', np.array(y_1))
    np.save(f'{saved_loaded_vecs_dir}/x_4_{num}_{SEED}.npy', np.array(features))

    return x_1, y_1, features


def read_test_data(samples_df):
    """
    Loading test features (from the .npy file) and labels into the df
    :return: features vector and label vector
    """

    # loading the features
    x_1 = []
    y_1 = []

    for row in samples_df.iterrows():
        x1_f = np.load(f'{vecs_dir}/{row[1]["file_name"]}.npy', allow_pickle=True).reshape([1, INPUT_SIZE])
        x_1.append(np.concatenate(x1_f[:]))
        label = samples_df[samples_df["file_name"] == row[1]["file_name"]]["label"].item()
        y_1.append(label)

    return x_1, y_1,


def train_CHAML(train_df, test_df):
    """
    Train CHAML model on train set
    :param train_df:
    :param test_df:
    :return:
    """
    # Load model
    model = CHAML_model()
    model.reset_states()

    # Load embedding vectors for query and support set.
    try:
        # Query embedding and label
        X1_tr = np.load(f'{saved_loaded_vecs_dir}/x_1_{SEED}_{SEED}.npy', allow_pickle=True)
        y_tr = np.load(f'{saved_loaded_vecs_dir}/y_1_{SEED}_{SEED}.npy', allow_pickle=True)
        # Support set (4 examples)
        X4_tr = np.load(f'{saved_loaded_vecs_dir}/x_4_{SEED}_{SEED}.npy', allow_pickle=True)

    except:
        X1_tr, y_tr, X4_tr = read_train_data(train_df, SEED)

    # Weighted loss
    weights = class_weight.compute_class_weight(class_weight='balanced',
                                                classes=np.unique(y_tr),
                                                y=y_tr)
    weights = {i: weights[i] for i in range(len(np.unique(y_tr)))}
    print(weights)

    lb = LabelEncoder()
    y_tr = to_categorical(lb.fit_transform(y_tr))

    X1_tr = np.asarray([np.asarray(item) for item in X1_tr])
    y_tr = np.asarray([np.asarray(item) for item in y_tr])
    X4_tr = np.asarray([np.asarray(item) for item in X4_tr])

    f1 = X4_tr[:, 0:INPUT_SIZE]
    f2 = X4_tr[:, INPUT_SIZE:INPUT_SIZE * 2]
    f3 = X4_tr[:, INPUT_SIZE * 2:INPUT_SIZE * 3]
    f4 = X4_tr[:, INPUT_SIZE * 3:INPUT_SIZE * 4]

    model.fit([X1_tr, f1, f2, f3, f4], y_tr,
              batch_size=TRAIN_BATCH_SIZE,
              epochs=EPOCHS, class_weight=weights)

    # Prediction on test set
    prediction(test_df, model)


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

    print(f"\n\n############################## SEED {SEED} SCORES: ###############################")

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


def prediction(test_df, model):
    """
    Predict labels of query samples, given the support set.

    :param test_df:
    :param model: trained model
    :return: prediction scores
    """
    print("#################################### TEST TASKS ####################################")
    # test_df = preprocess_df(test_df)
    test_df = test_df.groupby("task")

    test_labels = []
    test_preds = []

    # For each task:
    for name, group in test_df:
        print(f"----------------------------------------------{name}----------------------------------------------")
        # Order the support set as (True, True, False, False)
        support_ts_set = group[group['set'] == 'support'].sort_values('label')

        # mixed once to all
        # support_ts_set = group[group['set'] == 'support'].sample(frac=1, random_state=SEED).reset_index(drop=True)

        x_2, y_2 = read_test_data(support_ts_set.copy())
        x_2 = np.asarray(x_2).reshape([1, INPUT_SIZE * 4])
        print(support_ts_set)

        # Load query set
        query_ts_set = group[group['set'] == 'query'].sample(frac=1, random_state=SEED).reset_index(drop=True)
        amount_of_qr_samples = len(query_ts_set)

        # Add to each query sample, the support set of the current task
        df_preds = pd.DataFrame(columns=['1_features', '1_label', '4_features', '4_label'])
        for i in range(amount_of_qr_samples):
            sample = query_ts_set.iloc[[i]]
            x_1, y_1 = read_test_data(sample.copy())
            y_1 = y_1[0]
            row = {'1_features': x_1, '1_label': y_1, '4_features': x_2, '4_label': y_2}
            df_preds = df_preds.append(row, ignore_index=True)

        data_1 = np.array([np.array(item).reshape(INPUT_SIZE) for item in df_preds['1_features']])
        data_2 = np.array([np.array(item).reshape(INPUT_SIZE * 4) for item in df_preds['4_features']])

        f1 = data_2[:, :INPUT_SIZE]
        f2 = data_2[:, INPUT_SIZE:INPUT_SIZE * 2]
        f3 = data_2[:, INPUT_SIZE * 2:INPUT_SIZE * 3]
        f4 = data_2[:, INPUT_SIZE * 3:INPUT_SIZE * 4]

        scores = model.predict([data_1, f1, f2, f3, f4], verbose=0)

        print(np.transpose(scores))
        y_pred = np.argmax(scores, axis=1)  # soft-max
        labels = np.asarray([item for item in df_preds['1_label']])

        # Print current task statistics
        print('y_preds', y_pred)
        print('y_labels', labels)

        print(confusion_matrix(labels, y_pred))
        print("Accuracy:", accuracy_score(labels, y_pred))
        print('Precision:', precision_score(labels, y_pred))
        print('Recall:', recall_score(labels, y_pred))
        print('F1-Score:', f1_score(labels, y_pred))

        test_labels.extend(labels)
        test_preds.extend(y_pred)

        all_labels.extend(labels)
        all_preds.extend(y_pred)

    print_scores(test_labels, test_preds)


if __name__ == "__main__":

    times = []
    for i in range(0, 30):
        print(i)
        global SEED
        SEED = i

        # Define seed for all libraries
        os.environ['PYTHONHASHSEED'] = str(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        tf.random.set_seed(SEED)

        start = datetime.now()

        # Start model training
        df_train = pd.read_csv(train_path).sample(frac=1, random_state=SEED).reset_index(drop=True)
        df_test = pd.read_csv(test_path)
        train_CHAML(df_train, df_test)

        time = datetime.now() - start
        print("TIME:", time)
        times.append(time)

        keras.backend.clear_session()
        gc.collect()

    print("***************************** Final Scores *****************************")

    print(ALL_ACCS)
    print(ALL_F1S)
    print(f"Mean Accuracy:{mean(ALL_ACCS)}")
    print(f"Mean Precisions:{mean(ALL_PRECISIONS)}")
    print(f"Mean Recall:{mean(ALL_RECALLS)}")
    print(f"Mean F1s:{mean(ALL_F1S)}")
    # Calculating average running time of all 30 executions
    avgTime = pd.to_datetime(pd.Series(times).values.astype('datetime64[D]')).mean()
    print(f"Mean Time:{avgTime}")

    f.close()
