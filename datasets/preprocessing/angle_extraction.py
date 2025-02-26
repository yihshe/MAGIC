import os
import glob
import xml.etree.ElementTree as ET

BASE_DIR = '/maps/ys611/MAGIC/data/raw/Wytham/'
SAFE_FOLDER = 'S2A_MSIL2A_20180629T112111_N0500_R037_T30UXC_20230828T055820.SAFE'
metadata_file = os.path.join(BASE_DIR, SAFE_FOLDER, 'GRANULE', 'L2A_T30UXC_A015764_20180629T112537', 'MTD_TL.xml')

tree = ET.parse(metadata_file)
root = tree.getroot()

# Extract sun angles
sun_zenith = float(root.find(".//Mean_Sun_Angle/ZENITH_ANGLE").text)
sun_azimuth = float(root.find(".//Mean_Sun_Angle/AZIMUTH_ANGLE").text)

# Extract viewing angles (Example: Band 4)
viewing_angles = root.findall(".//Mean_Viewing_Incidence_Angle")
for angle in viewing_angles:
    band_id = angle.attrib.get("bandId")  # Identify band
    if band_id == "4":  # Change this to your band of interest
        view_zenith = float(angle.find("ZENITH_ANGLE").text)
        view_azimuth = float(angle.find("AZIMUTH_ANGLE").text)
        break

# Compute relative azimuth angle
relative_azimuth = abs(sun_azimuth - view_azimuth)
if relative_azimuth > 180:
    relative_azimuth = 360 - relative_azimuth

# Assign to RTM parameters
tto = view_zenith
tts = sun_zenith
psi = relative_azimuth

print(f"tto (Observation Zenith Angle): {tto}")
print(f"tts (Sun Zenith Angle): {tts}")
print(f"psi (Relative Azimuth Angle): {psi}")
