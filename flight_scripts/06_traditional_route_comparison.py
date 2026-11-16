
from pathlib import Path
import time
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "traditional_route_comparison"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------
# 1. Load data
# ------------------------------------
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")

for frame in [train, val]:
    frame["route"] = (
        frame["origin_city"].astype(str)
        + " -> "
        + frame["destination_city"].astype(str)
    )

# Use the same sample for every model
train = train.sample(
    n=min(30000, len(train)),
    random_state=42
).reset_index(drop=True)

val = val.sample(
    n=min(5000, len(val)),
    random_state=42
).reset_index(drop=True)

y_train = train["median_eur"].to_numpy()
y_val = val["median_eur"].to_numpy()

seen_routes = set(train["route"])
seen_mask = val["route"].isin(seen_routes).to_numpy()
unseen_mask = ~seen_mask

print("\n===== DATASET =====")
print("Training:", len(train))
print("Validation:", len(val))
print("Seen-route validation rows:", seen_mask.sum())
print("Unseen-route validation rows:", unseen_mask.sum())

# ------------------------------------
# 2. Feature preparation
# ------------------------------------
def prepare_features(use_route):

    columns = ["origin_city", "destination_city"]

    if use_route:
        columns.append("route")

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

    X_train_cat = encoder.fit_transform(train[columns])
    X_val_cat = encoder.transform(val[columns])

    scaler = StandardScaler()

    numeric = ["year", "month"]

    X_train_num = scaler.fit_transform(train[numeric])
    X_val_num = scaler.transform(val[numeric])

    X_train = hstack(
        [X_train_cat, csr_matrix(X_train_num)],
        format="csr"
    )

    X_val = hstack(
        [X_val_cat, csr_matrix(X_val_num)],
        format="csr"
    )

    return X_train, X_val

# ------------------------------------
# 3. Evaluation function
# ------------------------------------
results = []

def evaluate(name, features, predictions, seconds):

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    seen_mae = (
        mean_absolute_error(
            y_val[seen_mask],
            predictions[seen_mask]
        )
        if seen_mask.sum() > 0 else np.nan
    )

    unseen_mae = (
        mean_absolute_error(
            y_val[unseen_mask],
            predictions[unseen_mask]
        )
        if unseen_mask.sum() > 0 else np.nan
    )

    results.append({
        "Model": name,
        "Features": features,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Seen_Route_MAE": seen_mae,
        "Unseen_Route_MAE": unseen_mae,
        "Seconds": seconds
    })

    print(
        f"{name} ({features}): "
        f"MAE={mae:.2f}, "
        f"RMSE={rmse:.2f}, "
        f"R2={r2:.4f}, "
        f"Seen MAE={seen_mae:.2f}, "
        f"Unseen MAE={unseen_mae:.2f}"
    )

# ------------------------------------
# 4. Run comparisons
# ------------------------------------
for use_route in [False, True]:

    feature_name = (
        "Route-Aware" if use_route else "Original"
    )

    X_train, X_val = prepare_features(use_route)

    models = {
        "Ridge": Ridge(
            alpha=0.1,
            solver="lsqr"
        ),
        "KNN": KNeighborsRegressor(
            n_neighbors=30,
            weights="distance",
            algorithm="brute",
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=30,
            min_samples_leaf=5,
            random_state=42
        )
    }

    for name, model in models.items():

        print(f"\nTraining {name} ({feature_name})...")

        start = time.time()

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        predictions = np.clip(predictions, 0, None)

        elapsed = time.time() - start

        evaluate(
            name,
            feature_name,
            predictions,
            elapsed
        )

# ------------------------------------
# 5. Save results
# ------------------------------------
results_df = pd.DataFrame(results).sort_values("MAE")

results_df.to_csv(
    OUT / "route_comparison.csv",
    index=False
)

print("\n===== FINAL COMPARISON =====")
print(results_df.to_string(index=False))

print("\nResults saved to:", OUT)
