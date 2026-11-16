
from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------
# 1. File paths
# -----------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "fares_by_month_cleaned.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. Load the cleaned dataset
# -----------------------------------
df = pd.read_csv(DATA_FILE)

print("\n===== ORIGINAL DATA =====")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# -----------------------------------
# 3. Feature engineering
# -----------------------------------
df["departure_month"] = pd.to_datetime(
    df["departure_month"],
    format="%Y-%m"
)

df["year"] = df["departure_month"].dt.year
df["month"] = df["departure_month"].dt.month

# Keep only features and target
features = [
    "origin_city",
    "destination_city",
    "year",
    "month"
]

target = "median_eur"

df = df[
    ["departure_month"] + features + [target]
].copy()

# -----------------------------------
# 4. Chronological data splitting
# -----------------------------------
train = df[
    df["departure_month"] < "2026-12-01"
].copy()

val = df[
    df["departure_month"] == "2026-12-01"
].copy()

test = df[
    df["departure_month"] == "2027-01-01"
].copy()

print("\n===== DATA SPLIT =====")
print("Training rows:", len(train))
print("Validation rows:", len(val))
print("Testing rows:", len(test))
print("Excluded February rows:",
      (df["departure_month"] == "2027-02-01").sum())

assert len(train) > 0
assert len(val) > 0
assert len(test) > 0

# -----------------------------------
# 5. Separate features and targets
# -----------------------------------
X_train = train[features]
X_val = val[features]
X_test = test[features]

y_train = train[target]
y_val = val[target]
y_test = test[target]

# -----------------------------------
# 6. Define preprocessing
# -----------------------------------
categorical_features = [
    "origin_city",
    "destination_city"
]

numerical_features = [
    "year",
    "month"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cities",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            categorical_features
        ),
        (
            "numbers",
            StandardScaler(),
            numerical_features
        )
    ],
    sparse_threshold=1.0
)

# Fit using TRAINING data only
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same transformations
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# -----------------------------------
# 7. Save reusable datasets
# -----------------------------------
train.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)

# Save fitted preprocessing object
joblib.dump(
    preprocessor,
    OUTPUT_DIR / "flight_preprocessor.joblib"
)

# -----------------------------------
# 8. Display results
# -----------------------------------
print("\n===== PREPROCESSING COMPLETE =====")

print("Training shape:", X_train_processed.shape)
print("Validation shape:", X_val_processed.shape)
print("Testing shape:", X_test_processed.shape)

print("\nTraining target statistics:")
print(y_train.describe())

print("\nValidation target statistics:")
print(y_val.describe())

print("\nTesting target statistics:")
print(y_test.describe())

print("\nMissing values:")
print(df.isna().sum())

print("\nFiles saved to:", OUTPUT_DIR)
