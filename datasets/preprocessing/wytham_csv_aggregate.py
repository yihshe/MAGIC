import csv
import os
import glob
import pandas as pd

BASE_DIR = "/maps/ys611/MAGIC/data/raw/wytham"
csv_s2_paths = glob.glob(os.path.join(BASE_DIR, "rasters_sentinel2_2018", "*.csv"))
csv_frm4veg_path = os.path.join(BASE_DIR, "csv_validation_data/FRM_Veg_Wytham_20180703_V2_extr_NEW.csv")
csv_s2_angle_path = os.path.join(BASE_DIR, "csv_validation_data/Sentinel2_2018_angles.csv")

# Get all the dates from the Sentinel-2 CSV file names
dates = [os.path.basename(csv_path).split("_extracted.")[0] for csv_path in csv_s2_paths]

# Get all the sample ids from the first CSV file in the field sample_id
df = pd.read_csv(csv_s2_paths[0])
sample_ids = df["sample_id"].unique()
