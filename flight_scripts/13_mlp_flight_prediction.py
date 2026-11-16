
from pathlib import Path
import copy
import random
import os

# Limit CPU threading before importing numerical libraries.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

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

# ----------------------------------------
# 1. Settings
# ----------------------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "mlp_flight_prediction"

OUT.mkdir(parents=True, exist_ok=True)

# Force CPU instead of Apple MPS.
DEVICE = torch.device("cpu")

print("Device:", DEVICE, flush=True)
print("PyTorch version:", torch.__version__, flush=True)

# ----------------------------------------
# 2. Load data
# ----------------------------------------
print("\nLoading data...", flush=True)

train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")

for df in (train, val):
    df["route"] = (
        df["origin_city"].astype(str)
        + " -> "
        + df["destination_city"].astype(str)
    )

print("Training rows:", len(train), flush=True)
print("Validation rows:", len(val), flush=True)

# ----------------------------------------
# 3. Encode city categories
# ----------------------------------------
def make_vocabulary(series):
    values = sorted(
        series.dropna().astype(str).unique()
    )

    return {
        value: index + 1
        for index, value in enumerate(values)
    }


origin_vocab = make_vocabulary(
    train["origin_city"]
)

destination_vocab = make_vocabulary(
    train["destination_city"]
)


def encode_categories(df):

    origins = (
        df["origin_city"]
        .astype(str)
        .map(origin_vocab)
        .fillna(0)
        .astype("int64")
        .to_numpy()
    )

    destinations = (
        df["destination_city"]
        .astype(str)
        .map(destination_vocab)
        .fillna(0)
        .astype("int64")
        .to_numpy()
    )

    return origins, destinations


train_origin, train_destination = (
    encode_categories(train)
)

val_origin, val_destination = (
    encode_categories(val)
)

print("Unique training origins:", len(origin_vocab))
print(
    "Unique training destinations:",
    len(destination_vocab)
)

# ----------------------------------------
# 4. Calendar features
# ----------------------------------------
def calendar_features(df):

    month = df["month"].to_numpy(
        dtype=float
    )

    year = df["year"].to_numpy(
        dtype=float
    )

    return np.column_stack([
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        year - 2026.0
    ]).astype("float32")


train_calendar = calendar_features(train)
val_calendar = calendar_features(val)

# Target normalization based on training only.
target_mean = float(
    train["median_eur"].mean()
)

target_std = float(
    train["median_eur"].std()
)

if target_std <= 0:
    raise ValueError(
        "Target standard deviation must be positive."
    )

y_train = train["median_eur"].to_numpy(
    dtype="float32"
)

y_val = val["median_eur"].to_numpy(
    dtype="float32"
)

y_train_scaled = (
    (y_train - target_mean) / target_std
).astype("float32")

y_val_scaled = (
    (y_val - target_mean) / target_std
).astype("float32")

# ----------------------------------------
# 5. Create PyTorch datasets
# ----------------------------------------
def make_dataset(
    origins,
    destinations,
    calendar,
    targets
):

    return TensorDataset(
        torch.tensor(
            origins,
            dtype=torch.long
        ),
        torch.tensor(
            destinations,
            dtype=torch.long
        ),
        torch.tensor(
            calendar,
            dtype=torch.float32
        ),
        torch.tensor(
            targets,
            dtype=torch.float32
        )
    )


train_ds = make_dataset(
    train_origin,
    train_destination,
    train_calendar,
    y_train_scaled
)

val_ds = make_dataset(
    val_origin,
    val_destination,
    val_calendar,
    y_val_scaled
)

train_loader = DataLoader(
    train_ds,
    batch_size=1024,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_ds,
    batch_size=2048,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

# ----------------------------------------
# 6. Define the MLP model
# ----------------------------------------
class FlightMLP(nn.Module):

    def __init__(
        self,
        n_origins,
        n_destinations
    ):

        super().__init__()

        self.origin_embedding = nn.Embedding(
            n_origins + 1,
            16
        )

        self.destination_embedding = nn.Embedding(
            n_destinations + 1,
            32
        )

        self.network = nn.Sequential(

            nn.Linear(51, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(
        self,
        origin,
        destination,
        calendar
    ):

        origin_emb = self.origin_embedding(
            origin
        )

        destination_emb = (
            self.destination_embedding(
                destination
            )
        )

        x = torch.cat(
            [
                origin_emb,
                destination_emb,
                calendar
            ],
            dim=1
        )

        return self.network(x).squeeze(1)


model = FlightMLP(
    len(origin_vocab),
    len(destination_vocab)
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.001
)

loss_fn = nn.MSELoss()

print("\nModel initialized successfully.", flush=True)

# ----------------------------------------
# 7. Training with early stopping
# ----------------------------------------
EPOCHS = 80
PATIENCE = 10

best_val_loss = float("inf")
best_state = None
best_epoch = 0
wait = 0

history = []

print("\n===== TRAINING MLP =====", flush=True)

for epoch in range(1, EPOCHS + 1):

    model.train()

    train_loss_sum = 0.0

    for (
        origin,
        destination,
        calendar,
        target
    ) in train_loader:

        origin = origin.to(DEVICE)
        destination = destination.to(DEVICE)
        calendar = calendar.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(
            origin,
            destination,
            calendar
        )

        loss = loss_fn(
            predictions,
            target
        )

        loss.backward()

        optimizer.step()

        train_loss_sum += (
            loss.item() * len(target)
        )

    train_loss = (
        train_loss_sum / len(train_ds)
    )

    # Validation
    model.eval()

    val_loss_sum = 0.0

    with torch.no_grad():

        for (
            origin,
            destination,
            calendar,
            target
        ) in val_loader:

            origin = origin.to(DEVICE)
            destination = destination.to(DEVICE)
            calendar = calendar.to(DEVICE)
            target = target.to(DEVICE)

            predictions = model(
                origin,
                destination,
                calendar
            )

            loss = loss_fn(
                predictions,
                target
            )

            val_loss_sum += (
                loss.item() * len(target)
            )

    val_loss = (
        val_loss_sum / len(val_ds)
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    print(
        f"Epoch {epoch:02d} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f}",
        flush=True
    )

    if val_loss < best_val_loss - 1e-5:

        best_val_loss = val_loss

        best_state = copy.deepcopy(
            model.state_dict()
        )

        best_epoch = epoch
        wait = 0

    else:
        wait += 1

    if wait >= PATIENCE:

        print(
            "\nEarly stopping triggered.",
            flush=True
        )

        break

# Restore best model.
if best_state is None:
    raise RuntimeError(
        "Training did not produce a valid model."
    )

model.load_state_dict(best_state)

print(
    "\nBest epoch:",
    best_epoch,
    flush=True
)

# ----------------------------------------
# 8. Generate predictions
# ----------------------------------------
def predict(loader):

    model.eval()
    outputs = []

    with torch.no_grad():

        for (
            origin,
            destination,
            calendar,
            _
        ) in loader:

            predictions = model(
                origin.to(DEVICE),
                destination.to(DEVICE),
                calendar.to(DEVICE)
            )

            outputs.append(
                predictions.cpu().numpy()
            )

    scaled = np.concatenate(outputs)

    predictions = (
        scaled * target_std + target_mean
    )

    return np.clip(
        predictions,
        0,
        None
    )


train_eval_loader = DataLoader(
    train_ds,
    batch_size=2048,
    shuffle=False,
    num_workers=0
)

train_pred = predict(train_eval_loader)
val_pred = predict(val_loader)

# ----------------------------------------
# 9. Evaluate model
# ----------------------------------------
def metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(
            actual,
            predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),
        "R2": r2_score(
            actual,
            predicted
        )
    }


train_metrics = metrics(
    y_train,
    train_pred
)

val_metrics = metrics(
    y_val,
    val_pred
)

seen_routes = set(train["route"])

seen_mask = val["route"].isin(
    seen_routes
).to_numpy()

unseen_mask = ~seen_mask

seen_mae = mean_absolute_error(
    y_val[seen_mask],
    val_pred[seen_mask]
)

unseen_mae = mean_absolute_error(
    y_val[unseen_mask],
    val_pred[unseen_mask]
)

print("\n===== MLP RESULTS =====", flush=True)

print(
    "Train MAE:",
    round(train_metrics["MAE"], 4)
)

print(
    "Validation MAE:",
    round(val_metrics["MAE"], 4)
)

print(
    "Validation RMSE:",
    round(val_metrics["RMSE"], 4)
)

print(
    "Validation R2:",
    round(val_metrics["R2"], 4)
)

print(
    "Seen-route MAE:",
    round(seen_mae, 4)
)

print(
    "Unseen-route MAE:",
    round(unseen_mae, 4)
)

print(
    "MAE generalization gap:",
    round(
        val_metrics["MAE"]
        - train_metrics["MAE"],
        4
    )
)

# ----------------------------------------
# 10. Save results
# ----------------------------------------
history_df = pd.DataFrame(history)

history_df.to_csv(
    OUT / "training_history.csv",
    index=False
)

pd.DataFrame({
    "actual": y_val,
    "predicted": val_pred,
    "seen_route": seen_mask
}).to_csv(
    OUT / "validation_predictions.csv",
    index=False
)

torch.save(
    model.state_dict(),
    OUT / "mlp_model.pt"
)

# ----------------------------------------
# 11. Plot training curve
# ----------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Training loss"
)

plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    label="Validation loss"
)

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best epoch: {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("Scaled MSE Loss")
plt.title("MLP Training and Validation Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "training_curve.png",
    dpi=200
)

plt.close()

print(
    "\nResults saved to:",
    OUT,
    flush=True
)

print(
    "\nStep 13 completed successfully!",
    flush=True
)
