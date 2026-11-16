
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------------
# 1. Paths
# -----------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "traditional_tuning"
MODELS_DIR = RESULTS_DIR / "saved_models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. Load data
# -----------------------------------
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "validation.csv")

preprocessor = joblib.load(
    DATA_DIR / "flight_preprocessor.joblib"
)

features = [
    "origin_city",
    "destination_city",
    "year",
    "month"
]

X_train = preprocessor.transform(train[features])
X_val = preprocessor.transform(val[features])

y_train = train["median_eur"].to_numpy()
y_val = val["median_eur"].to_numpy()

print("\n===== FULL DATASET =====")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)

# -----------------------------------
# 3. Define model configurations
# -----------------------------------
experiments = []

# Ridge configurations
for alpha in [0.1, 1.0, 10.0, 100.0]:
    experiments.append((
        f"Ridge_alpha_{alpha}",
        Ridge(alpha=alpha, solver="lsqr"),
        "full"
    ))

# Decision Tree configurations
for depth in [10, 20, 30]:
    for leaf in [5, 20]:
        experiments.append((
            f"Tree_depth_{depth}_leaf_{leaf}",
            DecisionTreeRegressor(
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=42
            ),
            "full"
        ))

# KNN configurations
for k in [5, 15, 30]:
    experiments.append((
        f"KNN_k_{k}",
        KNeighborsRegressor(
            n_neighbors=k,
            weights="distance",
            algorithm="brute",
            metric="euclidean",
            n_jobs=-1
        ),
        "subset"
    ))

# -----------------------------------
# 4. KNN training subset
# -----------------------------------
rng = np.random.default_rng(42)

knn_indices = rng.choice(
    len(y_train),
    size=min(30000, len(y_train)),
    replace=False
)

X_train_knn = X_train[knn_indices]
y_train_knn = y_train[knn_indices]

# -----------------------------------
# 5. Train and evaluate
# -----------------------------------
results = []

for name, model, training_type in experiments:

    print(f"\nTraining {name}...")

    if training_type == "subset":
        X_fit = X_train_knn
        y_fit = y_train_knn
    else:
        X_fit = X_train
        y_fit = y_train

    start = time.time()

    model.fit(X_fit, y_fit)
    predictions = model.predict(X_val)

    elapsed = time.time() - start

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )
    r2 = r2_score(y_val, predictions)

    results.append({
        "Model": name,
        "Training_Rows": len(y_fit),
        "Validation_Rows": len(y_val),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Time_Seconds": elapsed
    })

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")
    print(f"Time: {elapsed:.2f} seconds")

    joblib.dump(
        model,
        MODELS_DIR / f"{name}.joblib"
    )

# -----------------------------------
# 6. Comparison table
# -----------------------------------
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAE"
).reset_index(drop=True)

results_df.to_csv(
    RESULTS_DIR / "tuning_comparison.csv",
    index=False
)

print("\n===== TUNING RESULTS =====")
print(results_df.to_string(index=False))

# -----------------------------------
# 7. Best model for each family
# -----------------------------------
families = ["Ridge", "Tree", "KNN"]

best_models = []

for family in families:
    family_results = results_df[
        results_df["Model"].str.startswith(family)
    ]

    best_models.append(family_results.iloc[0])

best_df = pd.DataFrame(best_models)

best_df.to_csv(
    RESULTS_DIR / "best_traditional_models.csv",
    index=False
)

print("\n===== BEST MODEL PER FAMILY =====")
print(best_df.to_string(index=False))

# -----------------------------------
# 8. Visualization
# -----------------------------------
plt.figure(figsize=(10, 5))

plt.bar(
    best_df["Model"],
    best_df["MAE"]
)

plt.title("Best Tuned Traditional Models")
plt.xlabel("Model")
plt.ylabel("Validation MAE (EUR)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "best_models_mae.png",
    dpi=200
)

plt.close()

print("\n===== COMPLETE =====")
print("Results saved to:", RESULTS_DIR)
