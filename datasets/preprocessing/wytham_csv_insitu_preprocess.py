# Reshape the in-situ measurements CSV file to have the same format as the aggregated Sentinel-2 CSV file.
import os
import numpy as np
import pandas as pd
import glob

S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
ATTRS = ['sample_id', 'date']

BASE_DIR = "/maps/ys611/MAGIC/data/raw/wytham"
SAVE_DIR = os.path.join("/maps/ys611/MAGIC/data/processed/rtm/wytham")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

csv_s2_paths = glob.glob(os.path.join(BASE_DIR, "rasters_sentinel2_2018", "*.csv"))
dates = [os.path.basename(csv_path).split("_extracted.")[0] for csv_path in csv_s2_paths]

csv_frm4veg_path = os.path.join(BASE_DIR, "csv_in_situ_validation/FRM_Veg_Wytham_20180703_V2_extr_NEW.csv")

# Read the CSV files seprated by ";"
csv_frm4veg = pd.read_csv(csv_frm4veg_path, delimiter=';')

# Create the columns of spectra to read from the in-situ CSV file
# The field is in a format like 'S2_20180606_B12' for all dates and all bands
# Reshape it in a way that all bands will remain in the field name, but dates will be in the rows

bands_columns = []
for date in dates:
    bands_columns.extend([f"S2_{date.replace("-", "")}_{band.split('_')[0]}" for band in S2_BANDS])

# Get all the plot labels to iterate over, if the plot label starts with 'E'
plot_labels = [plot_label for plot_label in csv_frm4veg["Plot Label"].unique() if 'E' in plot_label]








