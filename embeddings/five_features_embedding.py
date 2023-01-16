"""
Extract five features from audio samples spllited to train-test and save all embeddings as npy file in a directory
 named "fsfm_dir".
"""

import os

import librosa
import librosa.display

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

# Use GPU and define setting for getting reproducible results
os.environ['TF_DETERMINISTIC_OPS'] = '1'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Define seed for current task
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Define csv files paths
root_path = "../"
train_path = root_path + "train_df.csv"
test_path = root_path + "test_df.csv"

# Define the directory for the saves files
vecs_dir = "../fsfm_dir2_here"


def extract_features(files):
    """
    Extract the five features for each of the audio samples.
    :param files: audio samples.
    :return: five audio features.
    """

    file_name = files.path.replace("C:\\Users\\noaai\\Desktop\\new_claims\\all_good_folders\\meta_learning\\",
                        "..\\data\\train_test_division\\")
    file_name = os.path.join(str(file_name))

    # file_name = os.path.join(str(files.path))

    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = 0

    try:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                                  sr=sample_rate).T, axis=0)
    except:
        print(file_name)
    print(files.file_name)
    return mfccs, chroma, mel, contrast, tonnetz, files.file_name


def create_embedding(df):
    """
    Create the five sound embedding for the given data-frame.
    :param df: data frame of train/test.
    :return: the embedding.
    """

    features_label = df.apply(extract_features, axis=1)
    file_name = "features"
    # Saving the numpy array because it takes a long time to extract the features
    np.save(file_name, features_label)

    # loading the features
    features_label = np.load(file_name + '.npy', allow_pickle=True)
    # We create an empty list where we will concatenate all the features into one long feature
    # for each file to feed into our neural network
    features = []
    names = []
    for i in range(0, len(features_label)):
        try:
            features.append(np.concatenate((features_label[i][0], features_label[i][1],
                                            features_label[i][2], features_label[i][3],
                                            features_label[i][4]), axis=0))
            names.append(features_label[i][5])
        except:

            print("feature ", i, "didnt work")

    # Setting our X as a numpy array to feed into the neural network
    X = np.array(features)

    return X, names


def preprocess_df():
    """
    Call the extraction function, rescale the embeddings for the given set and save all to files.
    """

    train_set = df_train.copy()
    test_set = df_test.copy()
    ss = StandardScaler()

    # Train Set
    X,names = create_embedding(train_set)
    X = ss.fit_transform(X)
    names = set["file_name"]

    for x, name in zip(X, names):
        np.save(f'{vecs_dir}/{name}.npy', x)

    # Test Set
    X,names = create_embedding(test_set)
    X = ss.transform(X)
    names = set["file_name"]

    for x, name in zip(X, names):
        np.save(f'{vecs_dir}/{name}.npy', x)


if __name__ == "__main__":

    # Read train and test dataframes
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Start embedding process
    preprocess_df()
