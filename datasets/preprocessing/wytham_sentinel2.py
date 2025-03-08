import os
import re
import csv
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from rasterio.transform import xy

# 1) Read your shapefile using GeoPandas
shapefile_path = "/maps/ys611/MAGIC/data/raw/wytham/shapfile_wytham_woods_roi/perimeter_poly_with_clearings_region.shp"
gdf = gpd.read_file(shapefile_path)

# 2) List TIF files
raster_files = [
    "/maps/ys611/MAGIC/data/raw/wytham/rasters_sentinel2_2018/2018-06-06.tif",
    "/maps/ys611/MAGIC/data/raw/wytham/rasters_sentinel2_2018/2018-10-22.tif",
    # Add more paths if necessary...
]

# 3) Define a helper function to extract date from filename
#    This example looks for a pattern like YYYY-MM-DD in the file name.
def extract_date_from_path(raster_path):
    # Example: "2018-06-06.tif" -> "2018-06-06"
    match = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(raster_path))
    if match:
        return match.group(0)
    return "UnknownDate"

# 4) Define Sentinel-2 band names in the *exact order* of your TIF’s bands.
#    This is just an example for 4 bands (e.g., B2, B3, B4, B8). 
#    Adjust as needed to match your raster’s actual band count/ordering.
S2_FULL_BANDS = ['B01', 'B02_BLUE', 'B03_GREEN', 'B04_RED',
                'B05_RE1', 'B06_RE2', 'B07_RE3', 'B08_NIR1',
                'B8A_NIR2', 'B09_WV', 'B10', 'B11_SWI1',
                'B12_SWI2']
# If your TIF has more/less bands, update this list accordingly.

for raster_path in raster_files:
    # Extract date from filename
    date_str = extract_date_from_path(raster_path)

    with rasterio.open(raster_path) as src:
        # Ensure CRS matches shapefile
        if src.crs != gdf.crs:
            gdf = gdf.to_crs(src.crs)

        # 5) Mask the raster with the shapefile geometry
        shapes = gdf.geometry.tolist()
        out_image, out_transform = rasterio.mask.mask(
            src, shapes, crop=True, filled=True, invert=False, nodata=np.nan
        )

        # out_image now is a numpy array of shape (bands, height, width)
        bands, height, width = out_image.shape

        # 6) Save the masked raster for sanity check (visualization in QGIS)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": height,
            "width": width,
            "transform": out_transform
        })
        cropped_raster_path = raster_path.replace(".tif", "_cropped.tif")
        with rasterio.open(cropped_raster_path, "w", **out_meta) as dst:
            dst.write(out_image)

        # 7) Prepare to write CSV
        #    We want: [date, row, col, x, y, band1_name, band2_name, ...]
        csv_filename = raster_path.replace(".tif", "_extracted.csv")
        header = ["date", "row", "col", "x", "y"] + S2_FULL_BANDS

        with open(csv_filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            # 8) Iterate over each pixel (row, col)
            for row in range(height):
                for col in range(width):
                    # Extract pixel values across all bands
                    # out_image shape: (bands, height, width)
                    pixel_values = out_image[:, row, col]

                    # 8a) Check for nodata. 
                    #     If the nodata is 65535, we can test whether *all* bands are 65535
                    #     or just the first band. Adjust logic to your needs:
                    # if all(val == 65535 for val in pixel_values):
                    #     # Skip if this entire pixel is nodata
                    #     continue
                    if pixel_values[0] == np.nan:
                        # Skip if this pixel is nodata
                        continue

                    # 8b) Convert (row, col) to geospatial coords (x,y) if desired
                    x, y = xy(out_transform, row, col)

                    # 8c) Build the output row
                    #     Convert pixel_values to a list (e.g. float)
                    pixel_values_list = pixel_values.tolist()
                    # If it's a masked array, you might do .filled(np.nan) or similar.

                    row_data = [
                        date_str,
                        row,
                        col,
                        x,
                        y
                    ] + pixel_values_list

                    writer.writerow(row_data)

    print(f"Finished processing {raster_path}")
    print(f"- Cropped raster saved to {cropped_raster_path}")
    print(f"- CSV of pixel values saved to {csv_filename}\n")
