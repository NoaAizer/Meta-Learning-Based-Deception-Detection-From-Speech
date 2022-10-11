"""
Running FSFM model on given embedding vector.
Train-Test split by seed "2210"

Iterates over 30 seeds and return the averaged scores.
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout
from keras.models import Sequential

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils import compute_sample_weight, class_weight

from statistics import mean

# Use GPU and define setting for getting reproducible results
os.environ['TF_DETERMINISTIC_OPS'] = '1'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Define Batch-size and epochs
BATCH_SIZE = 256
EPOCHS = 50

# If Wav2Vec 2.0 embedding, otherwise five-sound-features
WAV2VEC = True

all_labels = []
all_preds = []

ALL_ACCS = []
ALL_RECALLS = []
ALL_PRECISIONS = []
ALL_F1S = []

# Define csv files paths
root_path = "../"
train_path = root_path + "train_df.csv"
test_path = root_path + "test_df.csv"

if WAV2VEC:
    INPUT_SIZE = 1024
    vecs_dir = "../wav2vec_vecs"
    file_suffix = "wav2vec"

else:
    INPUT_SIZE = 193
    vecs_dir = "../fsfm_dir"
    file_suffix = "five_fe"

# Write results to file
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
f = open(f"../outputs/output_FSFM_{file_suffix}.txt", 'w')
sys.stdout = f


def FSFM_model():
    model = Sequential()

    model.add(Dense(INPUT_SIZE, input_shape=(INPUT_SIZE,), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam', )

    return model


def data_preprocess(df):
    """
    Create X (features) and y  (labels) for the given dataframe.\n
    Load X from .npy file, and y from the df.

    :param df: Train/Test dataframe.
    :return: X, y
    """

    # Load features from .npy files
    X = []
    for name in df["file_name"]:
        data = np.load(f'{vecs_dir}/{name}.npy', allow_pickle=True).reshape([1, INPUT_SIZE])
        X.append(np.concatenate(data))
    X = np.array(X)

    # Load label from the matched df column
    y = np.array(df["label"])
    # All labels are 1
    if sum(y) == len(y):
        # Manual one hot encoding all to 1
        y =[[0., 1.] for i in y]
    else:
        # One hot encoding
        lb = LabelEncoder()
        y = to_categorical(lb.fit_transform(y))
    return X, y


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


def prediction(df_test, model):
    """
    Predict labels of query samples of each test task (person).

    Then arrange all results together.

    :param df_test: Dataframe of test samples.
    :param model: Trained model.
    :return:
    """

    print("#################################### TEST TASKS ####################################")

    test_preds = []
    test_labels = []

    df_test = df_test.groupby("task")
    # For each task:
    for name, group in df_test:
        print(f"----------------------------------------------{name}----------------------------------------------")
        # Predict only for query set (ignore support set).
        query_ts_set = group[group['set'] == 'query'].sample(frac=1, random_state=SEED).reset_index(drop=True)
        X_ts, y_ts = data_preprocess(query_ts_set.copy())

        y_pred = model.predict(X_ts)
        y_pred = np.argmax(y_pred, axis=1)
        labels = np.argmax(y_ts, axis=1)

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


def trainFSFM(df_train, df_test):
    # Preprocess data
    X_train, y_train = data_preprocess(df_train.copy())
    X_val, y_val = data_preprocess(df_test.copy())

    # Load model
    model = FSFM_model()
    model.reset_states()

    # Weighted loss
    s_weights = compute_sample_weight(class_weight='balanced', y=df_train.label)
    # Weighted loss
    weights = class_weight.compute_class_weight(class_weight='balanced',
                                                classes=np.unique(df_train.label),
                                                y=df_train.label)
    weights = {i: weights[i] for i in range(len(np.unique(df_train.label)))}
    print(weights)

    # Train model
    print(len(X_train),len(y_train))
    model.fit(X_train, y_train, epochs=EPOCHS, sample_weight=s_weights, batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val))

    # Prediction on test set
    prediction(df_test, model)

    return


if __name__ == "__main__":
    for i in range(0, 30):
        print(i)
        global SEED
        SEED = i

        # Define seed for all libraries
        os.environ['PYTHONHASHSEED'] = str(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        random.seed(SEED)

        # Start model training
        df_train = pd.read_csv(train_path)
        df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
        df_test = pd.read_csv(test_path)

        trainFSFM(df_train, df_test)

        # tf.keras.backend.clear_session()
        # gc.collect()

    print("***************************** Final Scores *****************************")

    print(ALL_ACCS)
    print(ALL_F1S)
    print(f"Mean Accuracy:{mean(ALL_ACCS)}")
    print(f"Mean Precisions:{mean(ALL_PRECISIONS)}")
    print(f"Mean Recall:{mean(ALL_RECALLS)}")
    print(f"Mean F1s:{mean(ALL_F1S)}")

    f.close()
