
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ---------------------------------------
# 1. Paths and settings
# ---------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "final_traditional_test"
OUT.mkdir(parents=True, exist_ok=True)

# All models use the same training records.
# KNN can be computationally expensive.
KNN_NEIGHBORS = 30

# ---------------------------------------
# 2. Load frozen datasets
# ---------------------------------------
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")
test = pd.read_csv(DATA / "test.csv")

for frame in [train, val, test]:
    frame["route"] = (
        frame["origin_city"].astype(str)
        + " -> "
        + frame["destination_city"].astype(str)
    )

y_train = train["median_eur"].to_numpy()
y_val = val["median_eur"].to_numpy()
y_test = test["median_eur"].to_numpy()

seen_routes = set(train["route"])

print("\n===== DATASET =====")
print("Training:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

# ---------------------------------------
# 3. Prepare features
# ---------------------------------------
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

    X_test_cat = encoder.transform(
        test[categorical]
    )

    scaler = StandardScaler()
    numeric = ["year", "month"]

    X_train_num = scaler.fit_transform(
        train[numeric]
    )

    X_val_num = scaler.transform(
        val[numeric]
    )

    X_test_num = scaler.transform(
        test[numeric]
    )

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

    return X_train, X_val, X_test

# ---------------------------------------
# 4. Evaluation
# ---------------------------------------
results = []

def evaluate(model_name, feature_name, split_name,
             actual, predicted, routes, seconds):

    predicted = np.clip(predicted, 0, None)

    seen_mask = routes.isin(seen_routes).to_numpy()
    unseen_mask = ~seen_mask

    seen_mae = (
        mean_absolute_error(
            actual[seen_mask],
            predicted[seen_mask]
        )
        if seen_mask.any()
        else np.nan
    )

    unseen_mae = (
        mean_absolute_error(
            actual[unseen_mask],
            predicted[unseen_mask]
        )
        if unseen_mask.any()
        else np.nan
    )

    results.append({
        "Model": model_name,
        "Features": feature_name,
        "Split": split_name,
        "Rows": len(actual),
        "MAE": mean_absolute_error(
            actual, predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
        "Seen_Route_MAE": seen_mae,
        "Unseen_Route_MAE": unseen_mae,
        "Seconds": seconds
    })

# ---------------------------------------
# 5. Train all six configurations
# ---------------------------------------
for use_route in [False, True]:

    feature_name = (
        "Route-Aware" if use_route else "Original"
    )

    print(f"\n===== {feature_name} FEATURES =====")

    X_train, X_val, X_test = prepare_features(
        use_route
    )

    models = {
        "Ridge": Ridge(
            alpha=0.15,
            solver="lsqr",
            tol=1e-6
        ),

        "KNN": KNeighborsRegressor(
            n_neighbors=KNN_NEIGHBORS,
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
        train_seconds = time.time() - start

        for split_name, X, y, frame in [
            ("Train", X_train, y_train, train),
            ("Validation", X_val, y_val, val),
            ("Test", X_test, y_test, test)
        ]:

            start_predict = time.time()
            predictions = model.predict(X)
            predict_seconds = time.time() - start_predict

            evaluate(
                name,
                feature_name,
                split_name,
                y,
                predictions,
                frame["route"],
                train_seconds + predict_seconds
            )

            score = results[-1]

            print(
                f"{split_name}: "
                f"MAE={score['MAE']:.2f}, "
                f"RMSE={score['RMSE']:.2f}, "
                f"R2={score['R2']:.4f}, "
                f"Seen={score['Seen_Route_MAE']:.2f}, "
                f"Unseen={score['Unseen_Route_MAE']:.2f}"
            )

# ---------------------------------------
# 6. Save results
# ---------------------------------------
df = pd.DataFrame(results)

df.to_csv(
    OUT / "all_traditional_results.csv",
    index=False
)

test_results = (
    df[df["Split"] == "Test"]
    .sort_values("MAE")
)

test_results.to_csv(
    OUT / "january_test_comparison.csv",
    index=False
)

# ---------------------------------------
# 7. Plot test MAE
# ---------------------------------------
labels = (
    test_results["Model"]
    + " ("
    + test_results["Features"]
    + ")"
)

plt.figure(figsize=(10, 5))
plt.bar(labels, test_results["MAE"])
plt.xticks(rotation=35, ha="right")
plt.ylabel("MAE (EUR)")
plt.title("January Test: Traditional Model Comparison")
plt.tight_layout()

plt.savefig(
    OUT / "january_test_mae.png",
    dpi=200
)

plt.close()

# ---------------------------------------
# 8. Print final comparison
# ---------------------------------------
print("\n===== JANUARY TEST COMPARISON =====")
print(
    test_results[
        [
            "Model",
            "Features",
            "MAE",
            "RMSE",
            "R2",
            "Seen_Route_MAE",
            "Unseen_Route_MAE"
        ]
    ].to_string(index=False)
)

print("\n===== COMPLETE =====")
print("Results saved to:", OUT)
