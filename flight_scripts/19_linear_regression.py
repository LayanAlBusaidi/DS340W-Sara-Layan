
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# 1. PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed"

OUT = ROOT / "results" / "linear_regression"

OUT.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. LOAD DATA
# ==================================================

train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")
test = pd.read_csv(DATA / "test.csv")

for df in [train, val, test]:
    df["route"] = (
        df["origin_city"].astype(str)
        + " -> "
        + df["destination_city"].astype(str)
    )

print("Training rows:", len(train))
print("Validation rows:", len(val))
print("Test rows:", len(test))

# ==================================================
# 3. FEATURE ENGINEERING
# ==================================================

categorical = [
    "origin_city",
    "destination_city",
    "route"
]

numeric = ["year", "month"]

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

X_train_cat = encoder.fit_transform(train[categorical])
X_val_cat = encoder.transform(val[categorical])
X_test_cat = encoder.transform(test[categorical])

scaler = StandardScaler()

X_train_num = scaler.fit_transform(train[numeric])
X_val_num = scaler.transform(val[numeric])
X_test_num = scaler.transform(test[numeric])

X_train = hstack(
    [X_train_cat, csr_matrix(X_train_num)],
    format="csr"
)

X_val = hstack(
    [X_val_cat, csr_matrix(X_val_num)],
    format="csr"
)

X_test = hstack(
    [X_test_cat, csr_matrix(X_test_num)],
    format="csr"
)

y_train = train["median_eur"].to_numpy()
y_val = val["median_eur"].to_numpy()
y_test = test["median_eur"].to_numpy()

# ==================================================
# 4. TRAIN ORDINARY LINEAR REGRESSION
# ==================================================

print("\nTraining Linear Regression...")

model = LinearRegression(
    fit_intercept=True,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete!")

# ==================================================
# 5. MAKE PREDICTIONS
# ==================================================

val_pred = np.maximum(model.predict(X_val), 0)
test_pred = np.maximum(model.predict(X_test), 0)

# ==================================================
# 6. EVALUATION FUNCTION
# ==================================================

def evaluate(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }

# ==================================================
# 7. FULL VALIDATION AND TEST RESULTS
# ==================================================

val_metrics = evaluate(y_val, val_pred)
test_metrics = evaluate(y_test, test_pred)

# ==================================================
# 8. SHARED DECEMBER SUBSET
# ==================================================

oct_routes = set(
    train.loc[
        (train["year"] == 2026)
        & (train["month"] == 10),
        "route"
    ]
)

nov_routes = set(
    train.loc[
        (train["year"] == 2026)
        & (train["month"] == 11),
        "route"
    ]
)

shared_routes = oct_routes & nov_routes

mask = (
    (val["year"] == 2026)
    & (val["month"] == 12)
    & val["route"].isin(shared_routes)
).to_numpy()

print("\nShared December records:", mask.sum())

if mask.sum() != 31058:
    raise ValueError(
        "Expected 31,058 records. "
        "Check the shared evaluation subset."
    )

shared_metrics = evaluate(
    y_val[mask],
    val_pred[mask]
)

# ==================================================
# 9. SAVE RESULTS
# ==================================================

results = pd.DataFrame([
    {
        "Model": "Linear Regression",
        "Dataset": "Full December Validation",
        "Rows": len(y_val),
        **val_metrics
    },
    {
        "Model": "Linear Regression",
        "Dataset": "January Test",
        "Rows": len(y_test),
        **test_metrics
    },
    {
        "Model": "Linear Regression",
        "Dataset": "Shared December Subset",
        "Rows": int(mask.sum()),
        **shared_metrics
    }
])

results.to_csv(
    OUT / "linear_regression_results.csv",
    index=False
)

predictions = val[
    ["origin_city", "destination_city", "year", "month"]
].copy()

predictions["actual"] = y_val
predictions["predicted"] = val_pred
predictions["shared_subset"] = mask

predictions.to_csv(
    OUT / "linear_regression_predictions.csv",
    index=False
)

# ==================================================
# 10. PRINT RESULTS
# ==================================================

print("\n========== LINEAR REGRESSION RESULTS ==========")

print(
    results.round(4).to_string(index=False)
)

print("\nFiles saved to:", OUT)

print("\nDONE!")

