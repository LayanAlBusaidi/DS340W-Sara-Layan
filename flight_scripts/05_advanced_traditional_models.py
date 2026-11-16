
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------------
# 1. File paths
# -----------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "advanced_traditional"

OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. Load datasets
# -----------------------------------
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")

# Create unique route identifier
for frame in [train, val]:
    frame["route"] = (
        frame["origin_city"]
        + " -> "
        + frame["destination_city"]
    )

# Sort BEFORE encoding to keep rows aligned
train = train.sort_values(
    "departure_month"
).reset_index(drop=True)

val = val.reset_index(drop=True)

target = "median_eur"

print("\n===== DATASET =====")
print("Training records:", len(train))
print("Validation records:", len(val))

# -----------------------------------
# 3. Route-aware one-hot encoding
# -----------------------------------
categorical_features = [
    "origin_city",
    "destination_city",
    "route"
]

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

encoder.fit(train[categorical_features])

X_train_cat = encoder.transform(
    train[categorical_features]
)

X_val_cat = encoder.transform(
    val[categorical_features]
)

print(
    "Categorical features:",
    X_train_cat.shape[1]
)

# Separate encoder without route
base_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

X_train_base_cat = base_encoder.fit_transform(
    train[["origin_city", "destination_city"]]
)

X_val_base_cat = base_encoder.transform(
    val[["origin_city", "destination_city"]]
)

# -----------------------------------
# 4. Historical price features
# -----------------------------------
groups = [
    "route",
    "origin_city",
    "destination_city"
]

history_columns = []

for group in groups:

    mean_col = f"{group}_past_mean"
    count_col = f"{group}_past_count"

    # Aggregate within each departure month
    monthly = (
        train.groupby(
            [group, "departure_month"]
        )[target]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("departure_month")
    )

    # Exclude the current month
    monthly["previous_sum"] = (
        monthly.groupby(group)["sum"].cumsum()
        - monthly["sum"]
    )

    monthly["previous_count"] = (
        monthly.groupby(group)["count"].cumsum()
        - monthly["count"]
    )

    monthly[mean_col] = (
        monthly["previous_sum"]
        / monthly["previous_count"].replace(
            0, np.nan
        )
    )

    monthly[count_col] = (
        monthly["previous_count"]
    )

    # Add historical features to training
    train = train.merge(
        monthly[
            [
                group,
                "departure_month",
                mean_col,
                count_col
            ]
        ],
        on=[group, "departure_month"],
        how="left",
        validate="many_to_one",
        sort=False
    )

    # Validation only uses training-period prices
    stats = train.groupby(group)[target].agg(
        ["mean", "count"]
    )

    val[mean_col] = val[group].map(
        stats["mean"]
    )

    val[count_col] = (
        val[group]
        .map(stats["count"])
        .fillna(0)
    )

    history_columns.extend(
        [mean_col, count_col]
    )

# -----------------------------------
# 5. Handle missing historical values
# -----------------------------------
# Zero means no historical price information.
# Count features help distinguish missing history.

for col in history_columns:
    train[col] = train[col].fillna(0)
    val[col] = val[col].fillna(0)

# -----------------------------------
# 6. Calendar features
# -----------------------------------
for frame in [train, val]:

    frame["month_sin"] = np.sin(
        2 * np.pi * frame["month"] / 12
    )

    frame["month_cos"] = np.cos(
        2 * np.pi * frame["month"] / 12
    )

numeric_columns = [
    "year",
    "month",
    "month_sin",
    "month_cos"
]

enhanced_columns = (
    numeric_columns + history_columns
)

# -----------------------------------
# 7. Prepare feature matrices
# -----------------------------------
def make_matrix(columns, use_route=True):

    scaler = StandardScaler()

    train_num = scaler.fit_transform(
        train[columns]
    )

    val_num = scaler.transform(
        val[columns]
    )

    if use_route:
        train_cat = X_train_cat
        val_cat = X_val_cat
    else:
        train_cat = X_train_base_cat
        val_cat = X_val_base_cat

    Xtr = hstack(
        [
            train_cat,
            csr_matrix(train_num)
        ],
        format="csr"
    )

    Xva = hstack(
        [
            val_cat,
            csr_matrix(val_num)
        ],
        format="csr"
    )

    return Xtr, Xva, scaler

# -----------------------------------
# 8. Define experiment configurations
# -----------------------------------
configurations = {
    "Original_Ridge": (
        ["year", "month"],
        False
    ),

    "Route_Ridge": (
        numeric_columns,
        True
    ),

    "Historical_Ridge": (
        enhanced_columns,
        False
    ),

    "Combined_Ridge": (
        enhanced_columns,
        True
    )
}

y_train = train[target].to_numpy()
y_val = val[target].to_numpy()

results = []

best_mae = float("inf")
best_details = None

# -----------------------------------
# 9. Train and tune Ridge models
# -----------------------------------
for config_name, (cols, use_route) in configurations.items():

    print(f"\n===== {config_name} =====")

    Xtr, Xva, scaler = make_matrix(
        cols,
        use_route
    )

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:

        for log_target in [False, True]:

            start = time.time()

            model = Ridge(
                alpha=alpha,
                solver="lsqr"
            )

            if log_target:

                model.fit(
                    Xtr,
                    np.log1p(y_train)
                )

                predictions = np.expm1(
                    model.predict(Xva)
                )

            else:

                model.fit(
                    Xtr,
                    y_train
                )

                predictions = model.predict(
                    Xva
                )

            predictions = np.clip(
                predictions,
                0,
                None
            )

            mae = mean_absolute_error(
                y_val,
                predictions
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_val,
                    predictions
                )
            )

            r2 = r2_score(
                y_val,
                predictions
            )

            elapsed = time.time() - start

            label = (
                f"{config_name}"
                f"_alpha{alpha}"
                f"_log{log_target}"
            )

            results.append({
                "Model": label,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Seconds": elapsed
            })

            print(
                f"alpha={alpha}, "
                f"log={log_target}: "
                f"MAE={mae:.2f}, "
                f"RMSE={rmse:.2f}, "
                f"R2={r2:.4f}"
            )

            if mae < best_mae:

                best_mae = mae

                best_details = {
                    "model": model,
                    "scaler": scaler,
                    "configuration": config_name,
                    "columns": cols,
                    "use_route": use_route,
                    "log_target": log_target,
                    "label": label
                }

# -----------------------------------
# 10. Historical route-price baseline
# -----------------------------------
baseline_predictions = (
    val["route_past_mean"].to_numpy()
)

baseline_mae = mean_absolute_error(
    y_val,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    y_val,
    baseline_predictions
)

results.append({
    "Model": "Historical_Route_Baseline",
    "MAE": baseline_mae,
    "RMSE": baseline_rmse,
    "R2": baseline_r2,
    "Seconds": 0
})

# -----------------------------------
# 11. Save comparison results
# -----------------------------------
results_df = pd.DataFrame(
    results
).sort_values(
    "MAE"
).reset_index(drop=True)

results_df.to_csv(
    OUT / "advanced_model_comparison.csv",
    index=False
)

# -----------------------------------
# 12. Save best Ridge model
# -----------------------------------
joblib.dump(
    {
        "best_model": best_details,
        "encoder": encoder,
        "base_encoder": base_encoder
    },
    OUT / "best_advanced_ridge.joblib"
)

# -----------------------------------
# 13. Print final results
# -----------------------------------
print("\n===== TOP 10 RESULTS =====")

print(
    results_df.head(10).to_string(
        index=False
    )
)

print("\n===== BEST MODEL =====")

print(
    results_df.iloc[0].to_string()
)

print("\n===== COMPLETE =====")
print("Results saved to:", OUT)
