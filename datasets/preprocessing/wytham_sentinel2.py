import os
import re
import csv
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from rasterio.transform import xy
import glob

BASE_DIR = "/maps/ys611/MAGIC/data/raw/wytham"

# Load the shapefile
shapefile_path = os.path.join(BASE_DIR, "shapfile_wytham_woods_roi", "perimeter_poly_with_clearings_region.shp")
gdf = gpd.read_file(shapefile_path)

# Load all Sentinel-2 rasters
raster_files = glob.glob(os.path.join(BASE_DIR, "rasters_sentinel2_2018", "*.tif"))

def extract_date_from_path(raster_path):
    # Example: "2018-06-06.tif" -> "2018-06-06"
    match = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(raster_path))
    if match:
        return match.group(0)
    return "UnknownDate"

S2_FULL_BANDS = ['B01', 'B02_BLUE', 'B03_GREEN', 'B04_RED',
                'B05_RE1', 'B06_RE2', 'B07_RE3', 'B08_NIR1',
                'B8A_NIR2', 'B09_WV', 'B10', 'B11_SWI1',
                'B12_SWI2']

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
            src, shapes, crop=True, filled=True, invert=False
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
        # Save the cropped raster to a new file
        # cropped_raster_path = raster_path.replace(".tif", "_cropped.tif")
        # with rasterio.open(cropped_raster_path, "w", **out_meta) as dst:
        #     dst.write(out_image)

        csv_filename = raster_path.replace(".tif", "_extracted.csv")
        header = ["date", "sample_id", "row", "col", "x", "y"]+S2_FULL_BANDS

        with open(csv_filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            # Iterate over each pixel (row, col)
            for row in range(height):
                for col in range(width):
                    # Extract pixel values across all bands
                    pixel_values = out_image[:, row, col]

                    # Skip if all bands are nodata
                    if pixel_values[0] == 65535:
                        # Skip if this pixel is nodata
                        continue

                    # unique sample id for each pixel based on row and col
                    sample_id = row * width + col

                    # Convert (row, col) to geospatial coords (x,y) if desired
                    x, y = xy(out_transform, row, col)

                    # Build the output row
                    pixel_values_list = pixel_values.tolist()

                    row_data = [date_str,sample_id,row,col,x,y] + pixel_values_list

                    writer.writerow(row_data)
    
    #TODO optional: subset the data to a smaller size for faster training
    print(f"Finished processing {raster_path}")
    # print(f"- Cropped raster saved to {cropped_raster_path}")
    print(f"- CSV of pixel values saved to {csv_filename}\n")
