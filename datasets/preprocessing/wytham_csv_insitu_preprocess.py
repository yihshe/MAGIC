# Reshape the in-situ measurements CSV file to have the same format as the aggregated Sentinel-2 CSV file.
import os
import numpy as np
import pandas as pd
import glob

S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
ATTRS = ['plot', 'class', 'sample_id', 'date', 'tto', 'tts', 'psi']

BASE_DIR = "/maps/ys611/MAGIC/data/raw/wytham"
SAVE_DIR = os.path.join(BASE_DIR, "csv_preprocessed_data")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

csv_s2_paths = glob.glob(os.path.join(BASE_DIR, "rasters_sentinel2_2018", "*.csv"))
csv_s2_rtm_angles_path = os.path.join(BASE_DIR, "csv_preprocessed_data/rasters_sentinel2_2018_rtm_angles.csv")

csv_s2_rtm_angles = pd.read_csv(csv_s2_rtm_angles_path)

dates = [os.path.basename(csv_path).split("_extracted.")[0] for csv_path in csv_s2_paths]
# sort the dates
dates.sort()
csv_frm4veg_path = os.path.join(BASE_DIR, "csv_in_situ_validation/FRM_Veg_Wytham_20180703_V2_extr_NEW.csv")

# Read the CSV files seprated by ";"
csv_frm4veg = pd.read_csv(csv_frm4veg_path, delimiter=';')

# Get all the plot labels to iterate over, if the plot label starts with 'E'
plot_labels = [plot_label for plot_label in csv_frm4veg["ESU.Label"].unique() if 'E' in plot_label]
plot_labels.sort()

# Create a new DataFrame to store the reshaped spectra, each row will be a plot label and a date
tree_class = 'unknown'
csv_frm4veg_s2_reshaped = pd.DataFrame(columns=ATTRS + S2_BANDS)
for i, plot_label in enumerate(plot_labels):
    for date in dates:
        # Get the Sentinel-2 bands for the plot label and date
        s2_per_plot_date = csv_frm4veg.loc[csv_frm4veg["ESU.Label"] == plot_label, [f"S2_{date.replace('-', '')}_{band.split('_')[0]}" for band in S2_BANDS]]
        tto = csv_s2_rtm_angles.loc[csv_s2_rtm_angles["date"] == date.replace('-', '.'), "tto"].values[0]
        tts = csv_s2_rtm_angles.loc[csv_s2_rtm_angles["date"] == date.replace('-', '.'), "tts"].values[0]
        psi = csv_s2_rtm_angles.loc[csv_s2_rtm_angles["date"] == date.replace('-', '.'), "psi"].values[0]
        # Create a new row with the plot label and date
        new_row = pd.DataFrame([[plot_label, tree_class, i, date.replace('-', '.'), tto, tts, psi] + s2_per_plot_date.values[0].tolist()], columns=ATTRS + S2_BANDS)
        # Append the new row to the reshaped DataFrame
        csv_frm4veg_s2_reshaped = pd.concat([csv_frm4veg_s2_reshaped, new_row], ignore_index=True)

# Save the reshaped DataFrame
csv_frm4veg_s2_reshaped.to_csv(os.path.join(SAVE_DIR, "frm4veg_sentinel2_2018.csv"), index=False)












