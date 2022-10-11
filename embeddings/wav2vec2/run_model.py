""" latest version of running wav2vec model on the cheat updated data
only train test without 5 fold"""

from model_functions import *
from model_classes import *

import pandas as pd
import numpy
import tensorflow as tf

numpy.set_printoptions(threshold=sys.maxsize)
torch.cuda.empty_cache()

from datasets import Dataset
from transformers import TrainingArguments

# Define seeds
SEED = 42
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

# Define csv files paths
root_path = "../../"
train_path = root_path + "train_df.csv"
test_path = root_path + "test_df.csv"

vecs_dir = root_path+ "wav2vec_vecs"

# Write details to file
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
f = open(f"../outputs/output_wav2vec2_ft.txt", 'w')
sys.stdout = f


def main():
    # Empty confusing matrix
    results = [[0, 0], [0, 0]]

    print(model_name_or_path)

    # ######################################## Prepare Data for Training #########################################

    train_df = pd.read_csv(train_path, delimiter=',')
    test_df = pd.read_csv(test_path, delimiter=',')

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(test_df)

    print(train_dataset)
    print(eval_dataset)

    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    config, feature_extractor, target_sampling_rate = prepare_data(num_labels)
    print(f"The target sampling rate: {target_sampling_rate}")

    # ######################################## Preprocess Data #########################################

    train_dataset = train_dataset.map(
        preprocess_function,
        batch_size=10,
        batched=True,
        num_proc=1
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batch_size=10,
        batched=True,
        num_proc=1
    )
    idx = 0

    print(f"Training labels: {train_dataset[idx]['labels']} - {train_dataset[idx]['label']}")

    # ######################################## Training (fine-tuning) #########################################
    print(train_dataset.features)
    data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
        tr_labels=train_dataset['label'],

    )
    model.freeze_feature_extractor()
    training_args = TrainingArguments(
        output_dir="wav2vec2-deception_detection",
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        gradient_accumulation_steps=3,
        evaluation_strategy="steps",
        num_train_epochs=5,
        load_best_model_at_end=False,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=1e-4,
        save_total_limit=1,
        do_eval=True,
        do_train=True,

    )
    print(training_args)

    wandb.init(name=training_args.output_dir, config=training_args)
    run_name = f'{model_name_or_path}_{training_args.num_train_epochs}_{training_args.gradient_accumulation_steps}_' \
               f'{training_args.per_device_train_batch_size}'
    wandb.init(name=run_name, reinit=True)

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor,
    )
    train_result = trainer.train()

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        print("########## PREDICT TRAIN #############")
        metrics = trainer.predict(train_dataset)
        res = metrics.predictions[1]
        vectors = [item for item in res]
        print(*vectors)  # print all train embedding vectors to be used later

        # Save each embedding as npy file
        for x, name in zip(vectors, train_df['file_name']):
            np.save(f'{vecs_dir}/{name}.npy', x)

        print("########## PREDICT TEST #############")
        metrics = trainer.predict(eval_dataset)
        res = metrics.predictions[1]
        vectors = [item for item in res]
        print(*vectors)  # print all test embedding vectors to be used later
        test_df["vec"] = vectors
        # Save each embedding as npy file
        for x, name in zip(vectors, test_df['file_name']):
            np.save(f'{vecs_dir}/{name}.npy', x)

        # ########################################################################
        label_names = [config.id2label[i] for i in range(config.num_labels)]
        y_true = [label_to_id(name, label_list) for name in eval_dataset["label"]]
        y_pred = np.argmax(metrics.predictions[0], axis=1)
        print(y_true)
        print(y_pred)

        labels = [0, 1]
        results += confusion_matrix(y_true, y_pred, labels=labels)
        print(classification_report(y_true, y_pred, target_names=label_names))

        wandb.log(
            {"prediction_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred,
                                                           class_names=label_names)})
        print(classification_report(y_true, y_pred, target_names=label_names))
        wandb.finish()

    wandb.log(
        {"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=labels_final, preds=pred_final,
                                                 class_names=label_list)})
    print(results)


if __name__ == "__main__":
    main()
    f.close()
