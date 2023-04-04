import pandas as pd
import numpy as np
import os

train_df = pd.read_csv("data/raw/train.csv")
train_df.to_csv("data/raw/train_full.csv", index=False)

patients = np.array(os.listdir("data/raw/train_images/")).astype(np.uint64)
train_df = train_df[train_df.patient_id.isin(patients)].reset_index().drop("index", axis=1)
train_df.to_csv("data/raw/train.csv", index=False)