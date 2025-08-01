import os
import numpy as np
import pandas as pd
import glob

ATTRS = ['date', 'tto', 'tts', 'psi']
BASE_DIR = "/maps/ys611/MAGIC/data/raw/wytham"
csv_s2_paths = glob.glob(os.path.join(BASE_DIR, "rasters_sentinel2_2018", "*.csv"))
dates = [os.path.basename(csv_path).split("_extracted.")[0] for csv_path in csv_s2_paths]
dates.sort()
csv_s2_angles_path = os.path.join(BASE_DIR, "csv_in_situ_validation/Sentinel2_2018_angles.csv")

# Read the CSV files seprated by ";"
csv_s2_angles = pd.read_csv(csv_s2_angles_path, delimiter=';')
csv_s2_rtm_angles = pd.DataFrame(columns=ATTRS)

for date in dates:
    # Given the row name and column name, get the value
    sun_zenith = csv_s2_angles.loc[csv_s2_angles["Angle_Name"] == 'SUN_ZENITH_ANGLE', date.replace('-','')].values[0]
    sun_azimuth = csv_s2_angles.loc[csv_s2_angles["Angle_Name"] == 'SUN_AZIMUTH_ANGLE', date.replace('-','')].values[0]
    view_zenith = csv_s2_angles.loc[csv_s2_angles["Angle_Name"] == 'Mean_Viewing_Incidence_Angle_B8_ZENITH_ANGLE', date.replace('-','')].values[0]
    view_azimuth = csv_s2_angles.loc[csv_s2_angles["Angle_Name"] == 'Mean_Viewing_Incidence_Angle_B8_AZIMUTH_ANGLE', date.replace('-','')].values[0]
    # Calculate the relative azimuth
    relative_azimuth = abs(sun_azimuth - view_azimuth)
    if relative_azimuth > 180:
        relative_azimuth = 360 - relative_azimuth
    # Assign to RTM parameters
    tto = view_zenith 
    tts = sun_zenith
    psi = relative_azimuth
    # Create a new row with the date and the RTM parameters
    new_row = pd.DataFrame([[date.replace('-', '.'), tto, tts, psi]], columns=ATTRS)
    # Append the new row to the reshaped DataFrame
    csv_s2_rtm_angles = pd.concat([csv_s2_rtm_angles, new_row], ignore_index=True)

# Save the reshaped DataFrame
csv_s2_rtm_angles.to_csv(os.path.join(BASE_DIR, "csv_preprocessed_data", "rasters_sentinel2_2018_rtm_angles.csv"), index=False)