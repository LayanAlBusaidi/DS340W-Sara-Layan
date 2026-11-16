
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor

# ---------------------------------------
# 1. Paths
# ---------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "tree_feature_engineering"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 2. Load data
# ---------------------------------------
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")

for frame in (train, val):
    frame["route"] = (
        frame["origin_city"].astype(str)
        + " -> "
        + frame["destination_city"].astype(str)
    )

y_train = train["median_eur"].to_numpy()
y_val = val["median_eur"].to_numpy()

seen_routes = set(train["route"])
seen_mask = val["route"].isin(seen_routes).to_numpy()
unseen_mask = ~seen_mask

print("Training rows:", len(train))
print("Validation rows:", len(val))
print("Seen validation routes:", seen_mask.sum())
print("Unseen validation routes:", unseen_mask.sum())

# ---------------------------------------
# 3. Frequency-encoded features
# ---------------------------------------
def frequency_features(train_df, val_df):

    train_x = train_df[["year", "month"]].copy()
    val_x = val_df[["year", "month"]].copy()

    for column in [
        "origin_city",
        "destination_city",
        "route"
    ]:
        counts = train_df[column].value_counts()

        train_x[f"{column}_frequency"] = (
            train_df[column]
            .map(counts)
            .fillna(0)
            .astype(float)
        )

        val_x[f"{column}_frequency"] = (
            val_df[column]
            .map(counts)
            .fillna(0)
            .astype(float)
        )

    return train_x, val_x


# ---------------------------------------
# 4. Native categorical features
# ---------------------------------------
def categorical_features(train_df, val_df):

    columns = [
        "origin_city",
        "destination_city",
        "route"
    ]

    train_x = train_df[
        ["year", "month"] + columns
    ].copy()

    val_x = val_df[
        ["year", "month"] + columns
    ].copy()

    for column in columns:

        # Categories are learned from training only.
        categories = pd.Index(
            train_df[column].dropna().unique()
        )

        dtype = pd.CategoricalDtype(
            categories=categories
        )

        train_x[column] = train_x[column].astype(dtype)
        val_x[column] = val_x[column].astype(dtype)

    return train_x, val_x


# ---------------------------------------
# 5. Evaluation
# ---------------------------------------
def evaluate(y, prediction):
    prediction = np.clip(prediction, 0, None)

    return {
        "MAE": mean_absolute_error(y, prediction),
        "RMSE": np.sqrt(
            mean_squared_error(y, prediction)
        ),
        "R2": r2_score(y, prediction)
    }


# ---------------------------------------
# 6. Experiments
# ---------------------------------------
experiments = [
    {
        "name": "RF Frequency",
        "features": "frequency",
        "model": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42
        )
    },
    {
        "name": "XGBoost Frequency",
        "features": "frequency",
        "model": XGBRegressor(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=42
        )
    },
    {
        "name": "XGBoost Native Categorical",
        "features": "categorical",
        "model": XGBRegressor(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            enable_categorical=True,
            max_cat_to_onehot=4,
            max_cat_threshold=64,
            n_jobs=-1,
            random_state=42
        )
    }
]

X_train_freq, X_val_freq = frequency_features(
    train, val
)

X_train_cat, X_val_cat = categorical_features(
    train, val
)

results = []

for experiment in experiments:

    name = experiment["name"]
    model = experiment["model"]

    if experiment["features"] == "frequency":
        X_train, X_val = X_train_freq, X_val_freq
    else:
        X_train, X_val = X_train_cat, X_val_cat

    print(f"\n===== {name} =====")

    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start

    train_pred = np.clip(
        model.predict(X_train), 0, None
    )

    val_pred = np.clip(
        model.predict(X_val), 0, None
    )

    train_metrics = evaluate(
        y_train, train_pred
    )

    val_metrics = evaluate(
        y_val, val_pred
    )

    seen_mae = (
        mean_absolute_error(
            y_val[seen_mask],
            val_pred[seen_mask]
        )
        if seen_mask.any()
        else np.nan
    )

    unseen_mae = (
        mean_absolute_error(
            y_val[unseen_mask],
            val_pred[unseen_mask]
        )
        if unseen_mask.any()
        else np.nan
    )

    results.append({
        "Model": name,
        "Train_MAE": train_metrics["MAE"],
        "Val_MAE": val_metrics["MAE"],
        "Train_RMSE": train_metrics["RMSE"],
        "Val_RMSE": val_metrics["RMSE"],
        "Train_R2": train_metrics["R2"],
        "Val_R2": val_metrics["R2"],
        "MAE_Gap": (
            val_metrics["MAE"]
            - train_metrics["MAE"]
        ),
        "Seen_Route_MAE": seen_mae,
        "Unseen_Route_MAE": unseen_mae,
        "Training_Seconds": seconds
    })

    print(
        f"Train MAE: {train_metrics['MAE']:.2f} | "
        f"Val MAE: {val_metrics['MAE']:.2f} | "
        f"Val R2: {val_metrics['R2']:.4f}"
    )

    print(
        f"Seen-route MAE: {seen_mae:.2f} | "
        f"Unseen-route MAE: {unseen_mae:.2f}"
    )

    print(f"Training time: {seconds:.1f}s")

# ---------------------------------------
# 7. Save results
# ---------------------------------------
df = pd.DataFrame(results).sort_values(
    "Val_MAE"
)

df.to_csv(
    OUT / "feature_engineering_comparison.csv",
    index=False
)

# ---------------------------------------
# 8. Plot comparison
# ---------------------------------------
plt.figure(figsize=(10, 5))

plt.bar(
    df["Model"],
    df["Val_MAE"]
)

plt.axhline(
    y=37.346380,
    linestyle="--",
    label="Route-Aware Ridge Benchmark"
)

plt.ylabel("Validation MAE (EUR)")
plt.title("Tree Feature Engineering Comparison")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "feature_engineering_mae.png",
    dpi=200
)

plt.close()

# ---------------------------------------
# 9. Final summary
# ---------------------------------------
print("\n===== FINAL FEATURE ENGINEERING COMPARISON =====")

print(
    df[
        [
            "Model",
            "Train_MAE",
            "Val_MAE",
            "Val_RMSE",
            "Val_R2",
            "MAE_Gap",
            "Seen_Route_MAE",
            "Unseen_Route_MAE"
        ]
    ].to_string(index=False)
)

print("\nResults saved to:", OUT)
