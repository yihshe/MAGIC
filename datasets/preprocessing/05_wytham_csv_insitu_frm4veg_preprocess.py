import os
import glob
import rasterio
import pandas as pd
import numpy as np
from datetime import datetime
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import rowcol
from openpyxl import load_workbook

BASE_DIR = "/maps/ys611/MAGIC/data/raw/wytham"
csv_frm4veg_path = os.path.join(BASE_DIR, "csv_in_situ_validation", "FRM_Veg_Wytham_20180703_V2_extr_NEW.csv")

SAVE_DIR = os.path.join(BASE_DIR, "csv_preprocessed_data")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

ATTRS = ['plot', 'sample_id', 'LAI_down', 'LAI_up', 'LAI', 'FCOVER_up', 'LCC', 'CCC']

ATTRS_FRM4VEG = ['LAI_down', 'LAI_up', 'LAI.1', 'FCOVER_up', 'LCC..g.m.2.', 'CCC..g.m.2.']

csv_frm4veg = pd.read_csv(csv_frm4veg_path, delimiter=';')
plot_labels = [plot_label for plot_label in csv_frm4veg["ESU.Label"].unique() if 'E' in plot_label]
plot_labels.sort()

csv_frm4veg_vars = pd.DataFrame(columns=ATTRS)

for i, plot_label in enumerate(plot_labels):
    frm4veg_vars = csv_frm4veg.loc[csv_frm4veg["ESU.Label"] == plot_label, ATTRS_FRM4VEG]
    csv_frm4veg.loc[csv_frm4veg["ESU.Label"] == plot_label, "sample_id"] = i
    new_row = pd.DataFrame([[plot_label, i, ]+frm4veg_vars.values[0].tolist()], columns=ATTRS)
    csv_frm4veg_vars = pd.concat([csv_frm4veg_vars, new_row], ignore_index=True)

# Save the reshaped DataFrame
csv_frm4veg_vars.to_csv(os.path.join(SAVE_DIR, "frm4veg_insitu.csv"), index=False)
print("Done!")