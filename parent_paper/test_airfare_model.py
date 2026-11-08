
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Allow imports from the original parent paper

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


from Scripts.Long_Sequence_Models.Bidirectional_LSTM_model import (
    Bidirectional_LSTM_model
)

# Set random seed
tf.keras.utils.set_random_seed(42)

# Load original preprocessed dataset

data_path = (
    ROOT / "preprocessed_data"
    / "flight_data_range_JFK_to_LAX_preprocessed.csv"
)


df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# Sort by travel date
df = df.sort_values(
    by=["Travel Day of Year", "Travel Hour"]
).reset_index(drop=True)

# Use 500 records for a quick test
sample = df.head(500)

# Separate features and target
X = sample.drop(columns=["Price ($)"]).to_numpy(dtype=np.float32)
y = sample["Price ($)"].to_numpy(dtype=np.float32)

# Chronological 80/10/10 split
X_train, X_val, X_test = X[:400], X[400:450], X[450:]
y_train, y_val, y_test = y[:400], y[400:450], y[450:]

# Add timestep dimension
X_train = np.expand_dims(X_train, axis=1)
X_val = np.expand_dims(X_val, axis=1)
X_test = np.expand_dims(X_test, axis=1)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# Build original Bidirectional LSTM
model = Bidirectional_LSTM_model(
    input_shape=(1, X_train.shape[2])
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)

# Train for 2 epochs
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=2,
    batch_size=32,
    verbose=1
)

# Evaluate on test data
predictions = model.predict(X_test, verbose=0).flatten()

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("\nTEST RESULTS")
print(f"MAE: ${mae:.2f}")
print(f"MSE: {mse:.2f}")

