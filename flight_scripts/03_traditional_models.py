
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------------
# 1. File paths
# -----------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "traditional_models"
MODELS_DIR = RESULTS_DIR / "saved_models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. Load datasets
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

target = "median_eur"

# -----------------------------------
# 3. Create reproducible pilot samples
# -----------------------------------
TRAIN_SIZE = 15000
VAL_SIZE = 2000
RANDOM_STATE = 42

train_sample = train.sample(
    n=min(TRAIN_SIZE, len(train)),
    random_state=RANDOM_STATE
)

val_sample = val.sample(
    n=min(VAL_SIZE, len(val)),
    random_state=RANDOM_STATE
)

X_train = preprocessor.transform(
    train_sample[features]
)
X_val = preprocessor.transform(
    val_sample[features]
)

y_train = train_sample[target]
y_val = val_sample[target]

print("\n===== PILOT DATA =====")
print("Training records:", len(y_train))
print("Validation records:", len(y_val))
print("Encoded features:", X_train.shape[1])

# -----------------------------------
# 4. Define models
# -----------------------------------
models = {
    "Mean Baseline": DummyRegressor(
        strategy="mean"
    ),

    "Linear Regression (Ridge)": Ridge(
        alpha=1.0,
        solver="lsqr"
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=5,
        weights="distance",
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=20,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )
}

# -----------------------------------
# 5. Train and evaluate models
# -----------------------------------
results = []

print("\n===== MODEL TRAINING =====")

for name, model in models.items():

    print(f"\nTraining: {name}")

    start_time = time.time()

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    elapsed = time.time() - start_time

    mae = mean_absolute_error(
        y_val, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    r2 = r2_score(
        y_val, predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Time_Seconds": elapsed
    })

    print(f"MAE: {mae:.2f} EUR")
    print(f"RMSE: {rmse:.2f} EUR")
    print(f"R2: {r2:.4f}")
    print(f"Time: {elapsed:.2f} seconds")

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    joblib.dump(
        model,
        MODELS_DIR / f"{filename}.joblib"
    )

# -----------------------------------
# 6. Create comparison table
# -----------------------------------
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE"
).reset_index(drop=True)

results_df.to_csv(
    RESULTS_DIR / "pilot_comparison.csv",
    index=False
)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# -----------------------------------
# 7. Plot model performance
# -----------------------------------
plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Model"],
    results_df["MAE"]
)

plt.title("Traditional ML Models - Pilot Validation MAE")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (EUR)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "pilot_mae_comparison.png",
    dpi=200
)

plt.close()

# -----------------------------------
# 8. Display best model
# -----------------------------------
best = results_df.iloc[0]

print("\n===== BEST PILOT MODEL =====")
print("Model:", best["Model"])
print(f"MAE: {best['MAE']:.2f} EUR")
print(f"RMSE: {best['RMSE']:.2f} EUR")
print(f"R2: {best['R2']:.4f}")

print("\nResults saved to:", RESULTS_DIR)
print("Training complete!")
