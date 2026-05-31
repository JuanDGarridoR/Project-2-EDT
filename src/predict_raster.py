# ==========================================
# PREDICT_RASTER.PY
# ==========================================

import joblib
import rasterio
import numpy as np

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading ANN model...\n")

model = joblib.load(
    "models/ann_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# ==========================================
# LOAD RASTER
# ==========================================

print("Loading raster...")

with rasterio.open(
    "data/sentinel_stack_crop.tif"
) as src:

    raster = src.read()

    profile = src.profile

# ==========================================
# RASTER INFO
# ==========================================

bands, rows, cols = raster.shape

print(f"Bands: {bands}")
print(f"Rows: {rows}")
print(f"Cols: {cols}")

# ==========================================
# RESHAPE
# ==========================================

pixels = raster.reshape(
    bands,
    rows * cols
).T

# ==========================================
# HANDLE NODATA
# ==========================================

valid_mask = np.all(
    pixels != 0,
    axis=1
)

valid_pixels = pixels[
    valid_mask
]

print(
    f"Valid pixels: {len(valid_pixels)}"
)

# ==========================================
# NORMALIZE
# ==========================================

valid_pixels_scaled = scaler.transform(
    valid_pixels
)

# ==========================================
# PREDICT
# ==========================================

print("Running ANN prediction...")

predictions = model.predict(
    valid_pixels_scaled
)

# ==========================================
# CLASS MAPPING
# ==========================================

classes = model.classes_

print("\nClasses detected:")

for i, cls in enumerate(classes):

    print(
        f"{cls} -> {i+1}"
    )

class_to_int = {

    cls: i + 1

    for i, cls in enumerate(classes)

}

pred_int = np.array(
    [
        class_to_int[p]
        for p in predictions
    ]
)

# ==========================================
# REBUILD RASTER
# ==========================================

output = np.zeros(
    rows * cols,
    dtype=np.uint8
)

output[
    valid_mask
] = pred_int

output = output.reshape(
    rows,
    cols
)

# ==========================================
# SAVE GEOTIFF
# ==========================================

profile.update(

    dtype=rasterio.uint8,

    count=1,

    compress="lzw"
)

output_path = (
    "outputs/final_classification.tif"
)

with rasterio.open(
    output_path,
    "w",
    **profile
) as dst:

    dst.write(
        output,
        1
    )

print("\n===================================")
print("CLASSIFICATION COMPLETED")
print("===================================")

print(
    f"\nGeoTIFF saved:\n{output_path}"
)