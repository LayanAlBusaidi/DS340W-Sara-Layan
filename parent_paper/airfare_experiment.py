
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from Scripts.Long_Sequence_Models.Bidirectional_LSTM_model import (
    Bidirectional_LSTM_model
)

# Set random seed
tf.keras.utils.set_random_seed(42)

# Load all preprocessed datasets
ROOT = Path(__file__).resolve().parent
data_dir = ROOT / "preprocessed_data"

files = sorted(data_dir.glob("*.csv"))
combined_df = pd.concat(
    [pd.read_csv(file) for file in files],
    ignore_index=True
)

combined_df = combined_df.sort_values(
    by=["Travel Day of Year", "Travel Hour"]
).reset_index(drop=True)

print("Total records:", len(combined_df))

# Create six-month windows
windows = []

for start_month in range(4, 7):
    end_month = start_month + 6

    window = combined_df[
        (combined_df["Travel Month"] >= start_month) &
        (combined_df["Travel Month"] < end_month)
    ].copy()

    windows.append(window)

    print(
        f"Window {len(windows)}: "
        f"Months {start_month}-{end_month-1}, "
        f"Records: {len(window)}"
    )

# Train first window
window = windows[0]

X = window.drop(columns=["Price ($)"]).to_numpy(dtype=np.float32)
y = window["Price ($)"].to_numpy(dtype=np.float32)

train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))

X_train = np.expand_dims(X[:train_size], axis=1)
X_val = np.expand_dims(
    X[train_size:train_size + val_size], axis=1
)
X_test = np.expand_dims(
    X[train_size + val_size:], axis=1
)

y_train = y[:train_size]
y_val = y[train_size:train_size + val_size]
y_test = y[train_size + val_size:]

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# Build original Bidirectional LSTM
model = Bidirectional_LSTM_model(
    input_shape=(1, X_train.shape[2])
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error"
)

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=25,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=2
)

# Evaluate
predictions = model.predict(X_test, verbose=0).flatten()

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("\nFINAL RESULTS")
print(f"MAE: ${mae:.2f}")
print(f"MSE: {mse:.2f}")

# Save results
results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)

pd.DataFrame({
    "Model": ["Bidirectional LSTM"],
    "Window": ["April-September"],
    "MAE": [mae],
    "MSE": [mse],
    "Epochs": [len(history.history["loss"])]
}).to_csv(results_dir / "airfare_results.csv", index=False)

print("Results saved to:", results_dir / "airfare_results.csv")
