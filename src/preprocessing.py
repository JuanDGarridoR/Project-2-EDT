# ==========================================
# PREPROCESSING.PY
# ==========================================

# IMPORT LIBRARIES
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...\n")

# Read CSV file
df = pd.read_csv(
    "data/project2_dataset_8bands.csv",
)

# Show first rows
print(df.head())

# ==========================================
# SELECT FEATURES
# ==========================================

# Spectral bands
X = df[
    [
        "BAND_1",
        "BAND_2",
        "BAND_3",
        "BAND_4",
        "BAND_5",
        "BAND_6",
        "BAND_7",
        "BAND_8",
    ]
]

# Target classes
y = df["class"]

# ==========================================
# NORMALIZATION
# ==========================================

print("\nNormalizing data...\n")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

print("\nSplitting dataset...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# ==========================================
# RESULTS
# ==========================================

print("Training samples:", len(X_train))

print("Testing samples:", len(X_test))

print("\nPreprocessing completed successfully.\n")