import argparse
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import TRAIN_DATA, PROCESSED_DATA_DIR


def split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray,
    train_df: pd.DataFrame,
    seed: int,
    n_splits: int = 1
) -> pd.DataFrame:
    """
    Split samples into train, test and val subsets using groups

    Args:
        X: dataframe with image indices
        y: dataframe with labels
        groups: groups for splitting dataframe
        train_df: initial dataframe
        seed: seed to fix randomness
        n_splits: number of splits

    Outputs:
        train_df: initial dataframe with info on splits
    """
    gss = GroupShuffleSplit(test_size=.1, n_splits=n_splits, random_state=seed)
    for train_index, test_index in gss.split(X, y, groups):
        train_df.loc[train_index, "subset"] = "train"
        train_df.loc[test_index, "subset"] = "val"
    return train_df


def upsample(
    train_df: pd.DataFrame,
    upsample_factor: int
) -> pd.DataFrame:
    """
    Upsample positive class

    Args:
        train_df: initial dataframe with samples
        upsample_factor: factor used to upsample positive class

    Outputs:
        train_df: upsampled dataframe
    """
    train_pos = train_df[(train_df['subset'] == "train") & (train_df['cancer'] == 1)]

    for i in range(upsample_factor):
        train_df = train_df.append(train_pos)

    train_df = train_df.reset_index().drop('index', axis=1)
    return train_df


def main() -> None:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--seed", type=int, default=13, help="Random seed")
    arg("--upsample_factor", type=int, default=10, help="Factor to upsample positive class in training dataset")
    args = parser.parse_args()

    train_df = pd.read_csv(TRAIN_DATA)
    X = train_df[['image_id']]
    y = train_df[['cancer']]
    groups = train_df['patient_id'].values

    train_df = split(X, y, groups, train_df, args.seed)
    train_df = upsample(train_df, args.upsample_factor)

    train_df.to_csv(PROCESSED_DATA_DIR / "data_subsets.csv",  index=False)


if __name__ == "__main__":
    main()
