
from pathlib import Path
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

# -----------------------------------
# 1. File paths
# -----------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "ridge_overfitting"

OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. Load data
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

# -----------------------------------
# 3. Identify seen and unseen routes
# -----------------------------------
seen_routes = set(train["route"])

seen_mask = val["route"].isin(seen_routes).to_numpy()
unseen_mask = ~seen_mask

print("\n===== DATASET =====")
print("Training rows:", len(train))
print("Validation rows:", len(val))
print("Seen-route validation rows:", seen_mask.sum())
print("Unseen-route validation rows:", unseen_mask.sum())

# -----------------------------------
# 4. Encode route-aware features
# -----------------------------------
categorical = [
    "origin_city",
    "destination_city",
    "route"
]

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

X_train_cat = encoder.fit_transform(train[categorical])
X_val_cat = encoder.transform(val[categorical])

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

print("Total features:", X_train.shape[1])

# -----------------------------------
# 5. Evaluate different alpha values
# -----------------------------------
alphas = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 10.0]

results = []

def metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }

for alpha in alphas:

    model = Ridge(
        alpha=alpha,
        solver="lsqr",
        tol=1e-6
    )

    model.fit(X_train, y_train)

    pred_train = np.clip(model.predict(X_train), 0, None)
    pred_val = np.clip(model.predict(X_val), 0, None)

    train_scores = metrics(y_train, pred_train)
    val_scores = metrics(y_val, pred_val)

    seen_mae = (
        mean_absolute_error(y_val[seen_mask], pred_val[seen_mask])
        if seen_mask.any() else np.nan
    )

    unseen_mae = (
        mean_absolute_error(y_val[unseen_mask], pred_val[unseen_mask])
        if unseen_mask.any() else np.nan
    )

    results.append({
        "Alpha": alpha,
        "Train_MAE": train_scores["MAE"],
        "Val_MAE": val_scores["MAE"],
        "Train_RMSE": train_scores["RMSE"],
        "Val_RMSE": val_scores["RMSE"],
        "Train_R2": train_scores["R2"],
        "Val_R2": val_scores["R2"],
        "MAE_Gap": val_scores["MAE"] - train_scores["MAE"],
        "Seen_MAE": seen_mae,
        "Unseen_MAE": unseen_mae
    })

    print(
        f"alpha={alpha:<5} | "
        f"Train MAE={train_scores['MAE']:.2f} | "
        f"Val MAE={val_scores['MAE']:.2f} | "
        f"Gap={val_scores['MAE'] - train_scores['MAE']:.2f} | "
        f"Seen={seen_mae:.2f} | "
        f"Unseen={unseen_mae:.2f}"
    )

# -----------------------------------
# 6. Save results
# -----------------------------------
df = pd.DataFrame(results).sort_values("Alpha")

df.to_csv(
    OUT / "ridge_overfitting_results.csv",
    index=False
)

# -----------------------------------
# 7. Plot training vs validation MAE
# -----------------------------------
plt.figure(figsize=(9, 5))

plt.plot(
    df["Alpha"],
    df["Train_MAE"],
    marker="o",
    label="Training MAE"
)

plt.plot(
    df["Alpha"],
    df["Val_MAE"],
    marker="o",
    label="Validation MAE"
)

plt.xscale("log")
plt.xlabel("Ridge Alpha (log scale)")
plt.ylabel("MAE (EUR)")
plt.title("Ridge Overfitting Check")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "train_vs_validation_mae.png",
    dpi=200
)

plt.close()

# -----------------------------------
# 8. Plot seen vs unseen route MAE
# -----------------------------------
plt.figure(figsize=(9, 5))

plt.plot(
    df["Alpha"],
    df["Seen_MAE"],
    marker="o",
    label="Seen Routes"
)

plt.plot(
    df["Alpha"],
    df["Unseen_MAE"],
    marker="o",
    label="Unseen Routes"
)

plt.xscale("log")
plt.xlabel("Ridge Alpha (log scale)")
plt.ylabel("MAE (EUR)")
plt.title("Seen vs Unseen Route Performance")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "seen_vs_unseen_mae.png",
    dpi=200
)

plt.close()

# -----------------------------------
# 9. Print final summary
# -----------------------------------
best = df.loc[df["Val_MAE"].idxmin()]

print("\n===== BEST VALIDATION MODEL =====")
print(best.to_string())

print("\n===== COMPLETE =====")
print("Results saved to:", OUT)
