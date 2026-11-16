
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor

# -----------------------------------
# 1. Paths
# -----------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "random_forest_xgboost"

OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. Load training and validation data
# -----------------------------------
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")

for frame in [train, val]:
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

# -----------------------------------
# 3. Feature preparation
# -----------------------------------
def prepare_features(use_route):

    categorical = [
        "origin_city",
        "destination_city"
    ]

    if use_route:
        categorical.append("route")

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

    X_train_cat = encoder.fit_transform(
        train[categorical]
    )

    X_val_cat = encoder.transform(
        val[categorical]
    )

    # Year and month remain numeric.
    from scipy.sparse import hstack, csr_matrix

    numeric = ["year", "month"]

    X_train_num = csr_matrix(
        train[numeric].to_numpy(dtype=float)
    )

    X_val_num = csr_matrix(
        val[numeric].to_numpy(dtype=float)
    )

    X_train = hstack(
        [X_train_cat, X_train_num],
        format="csr"
    )

    X_val = hstack(
        [X_val_cat, X_val_num],
        format="csr"
    )

    return X_train, X_val

# -----------------------------------
# 4. Metrics
# -----------------------------------
def calculate_metrics(y, predictions):

    predictions = np.clip(predictions, 0, None)

    return {
        "MAE": mean_absolute_error(y, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y, predictions)
        ),
        "R2": r2_score(y, predictions)
    }

# -----------------------------------
# 5. Models and hyperparameters
# -----------------------------------
model_configs = [
    (
        "Random Forest",
        {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": 42
        }
    ),
    (
        "Random Forest",
        {
            "n_estimators": 150,
            "max_depth": 25,
            "min_samples_leaf": 3,
            "n_jobs": -1,
            "random_state": 42
        }
    ),
    (
        "XGBoost",
        {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42
        }
    ),
    (
        "XGBoost",
        {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42
        }
    )
]

# -----------------------------------
# 6. Train and evaluate
# -----------------------------------
results = []

for use_route in [False, True]:

    feature_name = (
        "Route-Aware" if use_route else "Original"
    )

    print(f"\n===== {feature_name} FEATURES =====")

    X_train, X_val = prepare_features(use_route)

    for model_name, params in model_configs:

        print(f"\nTraining {model_name}: {params}")

        if model_name == "Random Forest":
            model = RandomForestRegressor(**params)
        else:
            model = XGBRegressor(**params)

        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        train_pred = np.clip(
            model.predict(X_train), 0, None
        )

        val_pred = np.clip(
            model.predict(X_val), 0, None
        )

        train_metrics = calculate_metrics(
            y_train, train_pred
        )

        val_metrics = calculate_metrics(
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
            "Model": model_name,
            "Features": feature_name,
            "Parameters": str(params),
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
            "Training_Seconds": elapsed
        })

        print(
            f"Train MAE: {train_metrics['MAE']:.2f} | "
            f"Val MAE: {val_metrics['MAE']:.2f} | "
            f"Val R2: {val_metrics['R2']:.4f} | "
            f"Seen: {seen_mae:.2f} | "
            f"Unseen: {unseen_mae:.2f} | "
            f"Time: {elapsed:.1f}s"
        )

# -----------------------------------
# 7. Save results
# -----------------------------------
df = pd.DataFrame(results).sort_values("Val_MAE")

df.to_csv(
    OUT / "rf_xgboost_comparison.csv",
    index=False
)

# -----------------------------------
# 8. Plot validation MAE
# -----------------------------------
labels = [
    f"{row.Model} ({row.Features}) #{i+1}"
    for i, row in enumerate(df.itertuples())
]

plt.figure(figsize=(11, 5))
plt.bar(labels, df["Val_MAE"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Validation MAE (EUR)")
plt.title("Random Forest and XGBoost Comparison")
plt.tight_layout()

plt.savefig(
    OUT / "rf_xgboost_validation_mae.png",
    dpi=200
)

plt.close()

# -----------------------------------
# 9. Final summary
# -----------------------------------
print("\n===== FINAL VALIDATION COMPARISON =====")

print(
    df[
        [
            "Model",
            "Features",
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

print("\n===== BEST MODEL =====")
print(df.iloc[0].to_string())

print("\nResults saved to:", OUT)
