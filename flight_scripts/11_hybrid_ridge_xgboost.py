
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor

# ----------------------------------------
# 1. Paths
# ----------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "hybrid_ridge_xgboost"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------
# 2. Load existing development data
# ----------------------------------------
train_all = pd.read_csv(DATA / "train.csv")
december = pd.read_csv(DATA / "validation.csv")

for frame in (train_all, december):
    frame["departure_month"] = pd.to_datetime(
        frame["departure_month"]
    )
    frame["route"] = (
        frame["origin_city"].astype(str)
        + " -> "
        + frame["destination_city"].astype(str)
    )

# Internal chronological split:
# September + October -> November
internal_train = train_all[
    train_all["departure_month"].dt.month.isin([9, 10])
].copy()

november = train_all[
    train_all["departure_month"].dt.month == 11
].copy()

assert len(internal_train) > 0
assert len(november) > 0

print("\n===== INTERNAL SPLIT =====")
print("September–October:", len(internal_train))
print("November:", len(november))
print("December:", len(december))

# ----------------------------------------
# 3. Ridge feature preparation
# ----------------------------------------
def ridge_features(train_df, eval_df):

    categorical = [
        "origin_city",
        "destination_city",
        "route"
    ]

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

    train_cat = encoder.fit_transform(
        train_df[categorical]
    )
    eval_cat = encoder.transform(
        eval_df[categorical]
    )

    scaler = StandardScaler()

    numeric = ["year", "month"]

    train_num = scaler.fit_transform(
        train_df[numeric]
    )
    eval_num = scaler.transform(
        eval_df[numeric]
    )

    X_train = hstack(
        [train_cat, csr_matrix(train_num)],
        format="csr"
    )
    X_eval = hstack(
        [eval_cat, csr_matrix(eval_num)],
        format="csr"
    )

    return X_train, X_eval

# ----------------------------------------
# 4. XGBoost categorical features
# ----------------------------------------
def xgb_features(train_df, eval_df):

    categorical = [
        "origin_city",
        "destination_city",
        "route"
    ]

    columns = ["year", "month"] + categorical

    X_train = train_df[columns].copy()
    X_eval = eval_df[columns].copy()

    for column in categorical:

        categories = pd.Index(
            train_df[column].dropna().unique()
        )

        dtype = pd.CategoricalDtype(
            categories=categories
        )

        X_train[column] = pd.Categorical(
            X_train[column],
            categories=categories
        )

        # Unknown evaluation categories become missing.
        X_eval[column] = pd.Categorical(
            X_eval[column].where(
                X_eval[column].isin(categories)
            ),
            categories=categories
        )

    return X_train, X_eval

# ----------------------------------------
# 5. Train both models
# ----------------------------------------
def train_and_predict(train_df, eval_df):

    y_train = train_df["median_eur"].to_numpy()

    X_train_ridge, X_eval_ridge = ridge_features(
        train_df, eval_df
    )

    ridge = Ridge(
        alpha=0.15,
        solver="lsqr",
        tol=1e-6
    )

    ridge.fit(X_train_ridge, y_train)

    ridge_pred = np.clip(
        ridge.predict(X_eval_ridge),
        0, None
    )

    X_train_xgb, X_eval_xgb = xgb_features(
        train_df, eval_df
    )

    xgb = XGBRegressor(
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

    xgb.fit(X_train_xgb, y_train)

    xgb_pred = np.clip(
        xgb.predict(X_eval_xgb),
        0, None
    )

    return ridge_pred, xgb_pred

# ----------------------------------------
# 6. Evaluation function
# ----------------------------------------
def evaluate(actual, predicted, seen_mask):

    predicted = np.clip(predicted, 0, None)
    unseen_mask = ~seen_mask

    return {
        "MAE": mean_absolute_error(
            actual, predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
        "Seen_MAE": (
            mean_absolute_error(
                actual[seen_mask],
                predicted[seen_mask]
            )
            if seen_mask.any() else np.nan
        ),
        "Unseen_MAE": (
            mean_absolute_error(
                actual[unseen_mask],
                predicted[unseen_mask]
            )
            if unseen_mask.any() else np.nan
        )
    }

# ----------------------------------------
# 7. Hybrid strategies
# ----------------------------------------
def predictions_for_strategy(
    strategy, ridge_pred, xgb_pred, seen_mask
):

    if strategy == "Ridge":
        return ridge_pred

    if strategy == "XGBoost":
        return xgb_pred

    if strategy == "Weighted 75% Ridge":
        return 0.75 * ridge_pred + 0.25 * xgb_pred

    if strategy == "Weighted 50% Ridge":
        return 0.50 * ridge_pred + 0.50 * xgb_pred

    if strategy == "Weighted 25% Ridge":
        return 0.25 * ridge_pred + 0.75 * xgb_pred

    if strategy == "Route-Based Hybrid":
        return np.where(
            seen_mask,
            ridge_pred,
            xgb_pred
        )

    raise ValueError(f"Unknown strategy: {strategy}")


strategies = [
    "Ridge",
    "XGBoost",
    "Weighted 75% Ridge",
    "Weighted 50% Ridge",
    "Weighted 25% Ridge",
    "Route-Based Hybrid"
]

# ----------------------------------------
# 8. November internal validation
# ----------------------------------------
print("\n===== NOVEMBER VALIDATION =====")

ridge_nov, xgb_nov = train_and_predict(
    internal_train, november
)

seen_nov = november["route"].isin(
    set(internal_train["route"])
).to_numpy()

y_nov = november["median_eur"].to_numpy()

nov_results = []

for strategy in strategies:

    pred = predictions_for_strategy(
        strategy,
        ridge_nov,
        xgb_nov,
        seen_nov
    )

    scores = evaluate(y_nov, pred, seen_nov)

    nov_results.append({
        "Strategy": strategy,
        **scores
    })

nov_df = pd.DataFrame(
    nov_results
).sort_values("MAE")

nov_df.to_csv(
    OUT / "november_hybrid_comparison.csv",
    index=False
)

print(
    nov_df.to_string(index=False)
)

best_strategy = nov_df.iloc[0]["Strategy"]

print("\nSelected strategy:", best_strategy)

# ----------------------------------------
# 9. December evaluation
# ----------------------------------------
print("\n===== DECEMBER EVALUATION =====")

ridge_dec, xgb_dec = train_and_predict(
    train_all, december
)

seen_dec = december["route"].isin(
    set(train_all["route"])
).to_numpy()

y_dec = december["median_eur"].to_numpy()

dec_results = []

for strategy in strategies:

    pred = predictions_for_strategy(
        strategy,
        ridge_dec,
        xgb_dec,
        seen_dec
    )

    scores = evaluate(y_dec, pred, seen_dec)

    dec_results.append({
        "Strategy": strategy,
        **scores
    })

dec_df = pd.DataFrame(
    dec_results
).sort_values("MAE")

dec_df.to_csv(
    OUT / "december_hybrid_comparison.csv",
    index=False
)

print(
    dec_df.to_string(index=False)
)

# ----------------------------------------
# 10. Plot comparisons
# ----------------------------------------
merged = nov_df[
    ["Strategy", "MAE"]
].merge(
    dec_df[["Strategy", "MAE"]],
    on="Strategy",
    suffixes=("_November", "_December")
)

x = np.arange(len(merged))
width = 0.35

plt.figure(figsize=(12, 5))

plt.bar(
    x - width / 2,
    merged["MAE_November"],
    width,
    label="November"
)

plt.bar(
    x + width / 2,
    merged["MAE_December"],
    width,
    label="December"
)

plt.xticks(
    x,
    merged["Strategy"],
    rotation=30,
    ha="right"
)

plt.ylabel("MAE (EUR)")
plt.title("Hybrid Model Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "hybrid_comparison.png",
    dpi=200
)

plt.close()

# ----------------------------------------
# 11. Final summary
# ----------------------------------------
selected_dec = dec_df[
    dec_df["Strategy"] == best_strategy
].iloc[0]

ridge_dec_mae = dec_df[
    dec_df["Strategy"] == "Ridge"
]["MAE"].iloc[0]

print("\n===== SELECTED HYBRID RESULT =====")
print("Selected using November:", best_strategy)
print(
    "December MAE:",
    round(selected_dec["MAE"], 4)
)
print(
    "December Ridge MAE:",
    round(ridge_dec_mae, 4)
)
print(
    "Difference vs Ridge:",
    round(
        selected_dec["MAE"] - ridge_dec_mae,
        4
    )
)

print("\nResults saved to:", OUT)
