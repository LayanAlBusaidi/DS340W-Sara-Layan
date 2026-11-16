
from pathlib import Path
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import copy
import random

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------
# 1. Settings
# ---------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

DEVICE = torch.device("cpu")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "recurrent_flight_models"
OUT.mkdir(parents=True, exist_ok=True)

EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 512

print("Device:", DEVICE, flush=True)

# ---------------------------------------
# 2. Load data
# ---------------------------------------
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")

df = pd.concat([train, val], ignore_index=True)

df["departure_month"] = pd.to_datetime(
    df["departure_month"]
).dt.to_period("M")

df["route"] = (
    df["origin_city"].astype(str)
    + " -> "
    + df["destination_city"].astype(str)
)

# Confirm each route-month is unique.
assert not df.duplicated(
    ["route", "departure_month"]
).any(), "Duplicate route-month records found."

# Route-month fare lookup.
fare_lookup = {
    (row.route, row.departure_month): row.median_eur
    for row in df.itertuples(index=False)
}

# ---------------------------------------
# 3. Construct sequences
# ---------------------------------------
def build_sequences(target_month):

    target_month = pd.Period(target_month, freq="M")

    target_rows = df[
        df["departure_month"] == target_month
    ]

    X = []
    y = []
    routes = []

    for row in target_rows.itertuples(index=False):

        previous_2 = (
            row.route,
            target_month - 2
        )

        previous_1 = (
            row.route,
            target_month - 1
        )

        if (
            previous_2 not in fare_lookup
            or previous_1 not in fare_lookup
        ):
            continue

        X.append([
            [fare_lookup[previous_2]],
            [fare_lookup[previous_1]]
        ])

        y.append(row.median_eur)
        routes.append(row.route)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        routes
    )


X_train, y_train, train_routes = (
    build_sequences("2026-11")
)

X_val, y_val, val_routes = (
    build_sequences("2026-12")
)

print("\n===== SEQUENCE DATA =====")
print("Training sequences:", len(X_train))
print("Validation sequences:", len(X_val))
print("Sequence shape:", X_train.shape)

assert len(X_train) > 0
assert len(X_val) > 0

# ---------------------------------------
# 4. Normalize using training only
# ---------------------------------------
# Inputs and targets are fare values.
# Fit normalization only on training records.

scale_mean = float(y_train.mean())
scale_std = float(y_train.std())

if scale_std <= 0:
    raise ValueError("Invalid training target scale.")

X_train_scaled = (
    (X_train - scale_mean) / scale_std
).astype(np.float32)

X_val_scaled = (
    (X_val - scale_mean) / scale_std
).astype(np.float32)

y_train_scaled = (
    (y_train - scale_mean) / scale_std
).astype(np.float32)

y_val_scaled = (
    (y_val - scale_mean) / scale_std
).astype(np.float32)

def make_loader(X, y, shuffle):

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )


train_loader = make_loader(
    X_train_scaled,
    y_train_scaled,
    True
)

val_loader = make_loader(
    X_val_scaled,
    y_val_scaled,
    False
)

# ---------------------------------------
# 5. Define recurrent network
# ---------------------------------------
class RecurrentFareModel(nn.Module):

    def __init__(self, model_type):
        super().__init__()

        hidden_size = 32

        if model_type == "RNN":
            self.recurrent = nn.RNN(
                input_size=1,
                hidden_size=hidden_size,
                batch_first=True
            )

        elif model_type == "GRU":
            self.recurrent = nn.GRU(
                input_size=1,
                hidden_size=hidden_size,
                batch_first=True
            )

        elif model_type == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                batch_first=True
            )

        else:
            raise ValueError("Unknown model type.")

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, _ = self.recurrent(x)

        last_step = output[:, -1, :]

        return self.head(
            last_step
        ).squeeze(1)

# ---------------------------------------
# 6. Evaluation
# ---------------------------------------
def evaluate(actual, predicted):

    predicted = np.clip(predicted, 0, None)

    return {
        "MAE": mean_absolute_error(
            actual, predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }


# Historical fare baselines.
baseline_last = X_val[:, -1, 0]
baseline_mean = X_val[:, :, 0].mean(axis=1)

results = [
    {
        "Model": "Last-Month Fare",
        **evaluate(y_val, baseline_last)
    },
    {
        "Model": "Two-Month Average",
        **evaluate(y_val, baseline_mean)
    }
]

# ---------------------------------------
# 7. Train each recurrent model
# ---------------------------------------
for model_type in ["RNN", "GRU", "LSTM"]:

    print(
        f"\n===== TRAINING {model_type} =====",
        flush=True
    )

    torch.manual_seed(SEED)

    model = RecurrentFareModel(
        model_type
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.001
    )

    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    history = []

    for epoch in range(1, EPOCHS + 1):

        model.train()
        train_loss_sum = 0.0

        for inputs, targets in train_loader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(inputs)

            loss = loss_fn(
                predictions,
                targets
            )

            loss.backward()
            optimizer.step()

            train_loss_sum += (
                loss.item() * len(targets)
            )

        train_loss = (
            train_loss_sum / len(X_train)
        )

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():

            for inputs, targets in val_loader:

                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                predictions = model(inputs)

                loss = loss_fn(
                    predictions,
                    targets
                )

                val_loss_sum += (
                    loss.item() * len(targets)
                )

        val_loss = (
            val_loss_sum / len(X_val)
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(
            f"{model_type} | "
            f"Epoch {epoch:02d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}",
            flush=True
        )

        if val_loss < best_loss - 1e-5:

            best_loss = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch
            wait = 0

        else:
            wait += 1

        if wait >= PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    # -----------------------------------
    # 8. Predictions
    # -----------------------------------
    model.eval()

    predictions = []

    with torch.no_grad():

        for inputs, _ in val_loader:

            output = model(
                inputs.to(DEVICE)
            )

            predictions.append(
                output.cpu().numpy()
            )

    val_pred = (
        np.concatenate(predictions)
        * scale_std
        + scale_mean
    )

    val_pred = np.clip(val_pred, 0, None)

    scores = evaluate(
        y_val,
        val_pred
    )

    results.append({
        "Model": model_type,
        **scores
    })

    print(
        f"\n{model_type} RESULTS: "
        f"MAE={scores['MAE']:.4f}, "
        f"RMSE={scores['RMSE']:.4f}, "
        f"R2={scores['R2']:.4f}"
    )

    print("Best epoch:", best_epoch)

    # Save training history.
    pd.DataFrame(history).to_csv(
        OUT / f"{model_type.lower()}_history.csv",
        index=False
    )

    # Save model weights.
    torch.save(
        model.state_dict(),
        OUT / f"{model_type.lower()}_model.pt"
    )

    # Save validation predictions.
    pd.DataFrame({
        "route": val_routes,
        "actual": y_val,
        "predicted": val_pred
    }).to_csv(
        OUT / f"{model_type.lower()}_predictions.csv",
        index=False
    )

# ---------------------------------------
# 9. Final comparison
# ---------------------------------------
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAE"
)

results_df.to_csv(
    OUT / "recurrent_comparison.csv",
    index=False
)

print("\n===== FINAL RECURRENT COMPARISON =====")

print(
    results_df.to_string(index=False)
)

# ---------------------------------------
# 10. Visualization
# ---------------------------------------
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["MAE"]
)

plt.ylabel("Validation MAE (EUR)")
plt.title(
    "RNN vs GRU vs LSTM and Historical Baselines"
)

plt.tight_layout()

plt.savefig(
    OUT / "recurrent_comparison.png",
    dpi=200
)

plt.close()

print("\nResults saved to:", OUT)
print("Step 14 completed successfully!")
