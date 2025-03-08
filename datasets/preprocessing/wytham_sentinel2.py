import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import csv

# 1. Read your shapefile using GeoPandas
shapefile_path = "/maps/ys611/MAGIC/data/raw/wytham/shapfile_wytham_woods_roi/perimeter_poly_with_clearings_region.shp"
gdf = gpd.read_file(shapefile_path)

# 2. List multiple TIF files if needed
raster_files = [
    "/maps/ys611/MAGIC/data/raw/wytham/rasters_sentinel2_2018/2018-06-06.tif",
    "/maps/ys611/MAGIC/data/raw/wytham/rasters_sentinel2_2018/2018-10-22.tif",
    # Add more paths if necessary...
]

for raster_path in raster_files:
    with rasterio.open(raster_path) as src:
        # (Optional) Reproject the GeoDataFrame if CRS doesn't match
        if src.crs != gdf.crs:
            gdf = gdf.to_crs(src.crs)

        # 3. Mask the raster with the shapefile geometry
        shapes = gdf.geometry.tolist()
        out_image, out_transform = rasterio.mask.mask(
            src, shapes, crop=True, filled=True, invert=False
        )
        # out_image is a numpy array of shape (bands, height, width)

        # 4. Optionally get and update metadata (if you need to save clipped rasters)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # 5. Flatten the masked array to extract pixel values
        bands, height, width = out_image.shape

        # Create row/col indices (rows first, cols second)
        rows, cols = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing='ij'  # ensures rows correspond to axis 0 and cols to axis 1
        )

        # Flatten them
        rows_flat = rows.ravel()
        cols_flat = cols.ravel()

        # Flatten the band data: shape becomes (bands, height*width)
        data_flat = out_image.reshape(bands, -1)

        # Convert (row, col) to geospatial coordinates using the transform
        xs, ys = rasterio.transform.xy(out_transform, rows_flat, cols_flat)

        # 6. Write to CSV
        # We'll store columns: x, y, band_1, band_2, ...
        csv_filename = raster_path.replace(".tif", "_extracted.csv")

        with open(csv_filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            header = ["x", "y"] + [f"band_{i+1}" for i in range(bands)]
            writer.writerow(header)

            # For each pixel
            for i in range(len(xs)):
                pixel_values = data_flat[:, i]  # all band values for the i-th pixel
                # If the pixel is masked (e.g., outside ROI), skip it or handle as needed
                if np.ma.is_masked(pixel_values):
                    continue  # skip masked
                # Convert to a regular list of floats
                pixel_values_list = pixel_values.tolist()
                row_data = [xs[i], ys[i]] + pixel_values_list
                writer.writerow(row_data)

    print(f"Extraction complete for {raster_path}, saved CSV to {csv_filename}")
