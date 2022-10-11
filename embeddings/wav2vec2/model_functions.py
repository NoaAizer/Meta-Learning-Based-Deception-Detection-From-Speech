"""
All functions related to wav2vec2 models

In this file you can define your Weights & Biases definitions.
"""

import os
import sys
import wandb
import torchaudio

from transformers import AutoConfig
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from model_classes import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Weights and Biases definitions
os.environ['WANDB_LOG_MODEL'] = 'true'
os.environ['WANDB_PROJECT'] = 'project_name'
wandb.login(key=sys.argv[1])

pred_final = []
labels_final = []
label_list = ['0', '1']

# Wav2Vec with fine-tuning on English task - path
model_name_or_path = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'

pooling_mode = "mean"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path, )
target_sampling_rate = feature_extractor.sampling_rate


def prepare_data(num_labels):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
    )

    setattr(config, 'pooling_mode', pooling_mode)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    target_sampling_rate = feature_extractor.sampling_rate

    return config, feature_extractor, target_sampling_rate


def speech_file_to_array_fn(path):
    path = os.path.join(str(path))
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy().flatten()
    return speech


def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label


def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples["path"]]
    target_list = [label_to_id(label, label_list) for label in examples["label"]]
    result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)
    return result


def compute_metrics(pred):
    label_idx = [0, 1]
    label_names = label_list

    preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = np.argmax(preds, axis=1)
    labels = pred.label_ids
    acc = (preds == pred.label_ids).astype(np.float32).mean().item()
    f1 = f1_score(labels, preds, average='macro')
    pred_final.append(preds)
    labels_final.append(labels)

    report = classification_report(y_true=labels, y_pred=preds, labels=label_idx, target_names=label_names)
    matrix = confusion_matrix(y_true=labels, y_pred=preds)
    print(report)
    print(matrix)

    wandb.log(
        {"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=preds, class_names=label_names)})

    return {"accuracy": acc, "f1_score": f1}


def get_vec_from_pre_trained(batch):
    """
       Get the audio vector embedding from the Wav2Vec model - no fine tuning.
       :param batch: batch of audio samples
       :return: audio embedding vectors of the given batch
       """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(model_name_or_path)
    setattr(config, 'pooling_mode', "mean")
    model = Wav2Vec2ForSpeechClassificationNoFineTune.from_pretrained(
        model_name_or_path,
        config=config,

    ).to(device)

    features = feature_extractor(batch["speech"], sampling_rate=feature_extractor.sampling_rate,
                                 return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    with torch.no_grad():
        res = model(input_values)
        res = res.tolist()
    batch["vec"] = res
    print(res)  # print the audio embedding vectors

    return batch


def get_vec_from_trained_model(batch):
    """
        Get the audio vector embedding from the Wav2Vec model - ** after ** fine-tuning.
        :param batch: batch of audio samples
        :return: audio embedding vectors of the given batch
        """
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to('cuda')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = feature_extractor(batch["speech"], sampling_rate=feature_extractor.sampling_rate,
                                 return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    print(input_values)
    with torch.no_grad():
        res = model(input_values).hidden_states
        res = res.tolist()
    batch["vec"] = res
    return batch
