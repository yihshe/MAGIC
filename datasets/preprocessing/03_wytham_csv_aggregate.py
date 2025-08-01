import os
import glob
import pandas as pd
import numpy as np


BASE_DIR = "/maps/ys611/MAGIC/data/raw/wytham"
csv_s2_paths = glob.glob(os.path.join(BASE_DIR, "rasters_sentinel2_2018", "*.csv"))
csv_s2_rtm_angles_path = os.path.join(BASE_DIR, "csv_preprocessed_data/rasters_sentinel2_2018_rtm_angles.csv")

csv_s2_rtm_angles = pd.read_csv(csv_s2_rtm_angles_path)

# Get all the dates from the Sentinel-2 CSV file names
dates = [os.path.basename(csv_path).split("_extracted.")[0] for csv_path in csv_s2_paths]

# Get all the sample ids from the first CSV file in the field sample_id
df0 = pd.read_csv(csv_s2_paths[0])
df0_sample_ids = df0["sample_id"].unique()

# Aggregate all csvs from csv_s2_paths into a single csv using pandas
# Also add the date column to the aggregated csv
df_list = []
for date, csv_path in zip(dates, csv_s2_paths):
    df = pd.read_csv(csv_path)
    assert np.array_equal(df['sample_id'].unique(), df0_sample_ids), "Sample IDs are not the same across all CSVs"
    # change date from e.g. 2018-06-29 to 2018.06.29
    df["date"] = date.replace("-", ".")
    df['class'] = 'unknown' # class of tree species is unknown
    for angle in ["tto", "tts", "psi"]:
        df[angle] = csv_s2_rtm_angles.loc[csv_s2_rtm_angles["date"] == date.replace("-", "."), angle].values[0]
    df_list.append(df)
df_s2 = pd.concat(df_list, axis=0)

# Save the aggregated CSV
# df_s2.to_csv(os.path.join(BASE_DIR, "csv_preprocessed_data", "rasters_sentinel2_2018.csv"), index=False)





