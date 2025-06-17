import csv
import os
import glob
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
import json

# Set the random seed for reproducibility
np.random.seed(42)

S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
ATTRS = ['sample_id', 'date']

BASE_DIR = "/maps/ys611/MAGIC/data/raw/wytham"
SAVE_DIR = os.path.join("/maps/ys611/MAGIC/data/processed/rtm/wytham/insitu_period_subset")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
SAVE_DIR_TRAIN = os.path.join(SAVE_DIR, "train.csv")
SAVE_DIR_VALID = os.path.join(SAVE_DIR, "valid.csv")
SAVE_DIR_TEST = os.path.join(SAVE_DIR, "test.csv")
SAVE_DIR_TEST_FRM4VEG = os.path.join(SAVE_DIR, "test_frm4veg.csv")

CSV_S2_2018_DIR = os.path.join(BASE_DIR, "csv_preprocessed_data", "rasters_sentinel2_2018.csv")
CSV_S2_FRM4VEG_DIR = os.path.join(BASE_DIR, "csv_preprocessed_data", "frm4veg_sentinel2_2018.csv")

# TODO add the S2 spectra from frm4veg sample sites and standardize the data
SAMPLE_RATIO = 0.1 #0.5 for insitu_period
SPLIT_RATIO = 0.2

csv_s2_2018 = pd.read_csv(CSV_S2_2018_DIR)

# Filter the data to only include the dates of interest
# dates = ['2018.06.26', '2018.06.29', '2018.07.06', '2018.07.11']
dates = ['2018.06.29', '2018.07.06']
csv_s2_2018 = csv_s2_2018[csv_s2_2018["date"].isin(dates)]

sample_ids = csv_s2_2018["sample_id"].unique()
# TODO sample a subset of the data based on the sample id and the sample ratio
sample_ids = np.random.choice(sample_ids, int(len(sample_ids)*SAMPLE_RATIO), replace=False)

# split the data according to the sample id
train_sample_ids, test_sample_ids = train_test_split(
    sample_ids, test_size=SPLIT_RATIO, random_state=42)
train_sample_ids, valid_sample_ids = train_test_split(
    train_sample_ids, test_size=SPLIT_RATIO, random_state=42)

# Standardize the train, valid and test sets based on S2_BANDS
def standardize(df, columns, scaler=None):
    factor = 10000.0
    df[columns] = df[columns]/factor
    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    return df, scaler

df_train = csv_s2_2018[csv_s2_2018["sample_id"].isin(train_sample_ids)]
df_valid = csv_s2_2018[csv_s2_2018["sample_id"].isin(valid_sample_ids)]
df_test = csv_s2_2018[csv_s2_2018["sample_id"].isin(test_sample_ids)]
df_test_frm4veg = pd.read_csv(CSV_S2_FRM4VEG_DIR)

scaler = None
df_train, scaler = standardize(df_train, S2_BANDS, scaler)
df_valid, _ = standardize(df_valid, S2_BANDS, scaler)
df_test, _ = standardize(df_test, S2_BANDS, scaler)
df_test_frm4veg, _ = standardize(df_test_frm4veg, S2_BANDS, scaler)

# Save the train, valid and test sets
df_train.to_csv(SAVE_DIR_TRAIN, index=False)
df_valid.to_csv(SAVE_DIR_VALID, index=False)
df_test.to_csv(SAVE_DIR_TEST, index=False)
df_test_frm4veg.to_csv(SAVE_DIR_TEST_FRM4VEG, index=False)

# Save the scaler
np.save(os.path.join(SAVE_DIR, "train_x_mean.npy"), scaler.mean_)
np.save(os.path.join(SAVE_DIR, "train_x_scale.npy"), scaler.scale_)
print("Done!")


