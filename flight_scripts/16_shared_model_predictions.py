
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# ---------------------------------------
# 1. Paths
# ---------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "parent_paper_comparison"

OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 2. Load datasets
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

# ---------------------------------------
# 3. Reproduce original feature encoding
# ---------------------------------------
categorical = [
    "origin_city",
    "destination_city",
    "route"
]

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

scaler = StandardScaler()
numeric = ["year", "month"]

X_train_num = scaler.fit_transform(
    train[numeric]
)

X_val_num = scaler.transform(
    val[numeric]
)

X_train = hstack(
    [X_train_cat, csr_matrix(X_train_num)],
    format="csr"
)

X_val = hstack(
    [X_val_cat, csr_matrix(X_val_num)],
    format="csr"
)

# ---------------------------------------
# 4. Train original configurations
# ---------------------------------------
models = {
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

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = np.clip(
        model.predict(X_val),
        0,
        None
    )

    filename = (
        name.lower().replace(" ", "_")
        + "_validation_predictions.csv"
    )

    output = pd.DataFrame({
        "route": val["route"].to_numpy(),
        "actual": y_val,
        "predicted": predictions
    })

    output.to_csv(
        OUT / filename,
        index=False
    )

    print(
        "Full December MAE:",
        mean_absolute_error(y_val, predictions)
    )
    print("Saved:", OUT / filename)

print("\nComplete!")
