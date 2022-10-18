import random
import numpy as np
import pandas as pd


def create_train_samples(df_train, seed=42):
    """
    Create data-frame from the train df, contains 1 sample and 4 samples pairs.
    :param seed:
    :param df_train: Original train data-frame
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    # Create data frame with name , first sample and the four-samples to be used later
    column_names = ['task', '1', '4']

    df = pd.DataFrame(columns=column_names)
    df_train = df_train.groupby("task")
    for name, group in df_train:

        only_2_false, only_2_true = False, False
        # There are only 2 samples of False

        if len(group[group['label'] == 1]) == 2:
            only_2_false = True
        # There are only 2 samples of True
        if len(group[group['label'] == 0]) == 2:
            only_2_true = True

        group.reset_index(drop=True, inplace=True)
        for i in range(len(group)):
            sample = group.loc[i]
            if only_2_false and str(sample["label"]) == "1":
                continue
            if only_2_true and str(sample["label"]) == "0":
                continue

            temp_gr = group.drop(i, inplace=False)

            # Sample 2 True & 2 False
            true_set = temp_gr[temp_gr['label'] == 0].sample(n=2)
            false_set = temp_gr[temp_gr['label'] == 1].sample(n=2)
            all_4_samples = true_set.append(false_set)
            row = {'task': name, '1_name': sample['file_name'], '4_name': all_4_samples["file_name"].tolist()}

            df = df.append(row, ignore_index=True)

    print(df.head())
    return df


if __name__ == "__main__":
    train_path = "train_df.csv"  # complete
    train_df = pd.read_csv(train_path)
    df = create_train_samples(train_df)
    df.to_csv("train_CHAML.csv", ignore_index=True)
