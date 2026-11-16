
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------
# 1. Paths
# ---------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

OUT = RESULTS / "fair_model_comparison"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 2. Load December validation data
# ---------------------------------------
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")

for df in (train, val):
    df["route"] = (
        df["origin_city"].astype(str)
        + " -> "
        + df["destination_city"].astype(str)
    )

    df["period"] = pd.to_datetime(
        df["departure_month"]
    ).dt.to_period("M")

# December rows with October and November
# fare observations for the same route.
oct_routes = set(
    train.loc[
        train["period"] == pd.Period("2026-10"),
        "route"
    ]
)

nov_routes = set(
    train.loc[
        train["period"] == pd.Period("2026-11"),
        "route"
    ]
)

eligible_routes = oct_routes & nov_routes

eligible = val[
    (val["period"] == pd.Period("2026-12"))
    & val["route"].isin(eligible_routes)
].copy()

print("Eligible December records:", len(eligible))

# Verify that route identifies one record
# in the December evaluation subset.
assert not eligible["route"].duplicated().any()

# ---------------------------------------
# 3. Model metrics
# ---------------------------------------
def evaluate(name, actual, predicted):

    return {
        "Model": name,
        "N": len(actual),
        "MAE": mean_absolute_error(
            actual, predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }


scores = []

# ---------------------------------------
# 4. Historical fare baselines
# ---------------------------------------
oct_fares = (
    train[
        train["period"] == pd.Period("2026-10")
    ]
    .set_index("route")["median_eur"]
)

nov_fares = (
    train[
        train["period"] == pd.Period("2026-11")
    ]
    .set_index("route")["median_eur"]
)

last_month = eligible["route"].map(
    nov_fares
).to_numpy()

two_month_average = (
    eligible["route"].map(oct_fares).to_numpy()
    + last_month
) / 2

actual = eligible["median_eur"].to_numpy()

scores.append(
    evaluate(
        "Last-Month Fare",
        actual,
        last_month
    )
)

scores.append(
    evaluate(
        "Two-Month Average",
        actual,
        two_month_average
    )
)

# ---------------------------------------
# 5. Load recurrent predictions
# ---------------------------------------
recurrent_dir = (
    RESULTS / "recurrent_flight_models"
)

for model in ["rnn", "gru", "lstm"]:

    path = recurrent_dir / (
        f"{model}_predictions.csv"
    )

    predictions = pd.read_csv(path)

    # Join using route rather than row position.
    if predictions["route"].duplicated().any():
        raise ValueError(
            f"Duplicate routes in {path}"
        )

    lookup = predictions.set_index(
        "route"
    )["predicted"]

    aligned = eligible["route"].map(lookup)

    if aligned.isna().any():
        raise ValueError(
            f"Missing {model} predictions."
        )

    scores.append(
        evaluate(
            model.upper(),
            actual,
            aligned.to_numpy()
        )
    )

# ---------------------------------------
# 6. Load MLP predictions
# ---------------------------------------
mlp_path = (
    RESULTS
    / "mlp_flight_prediction"
    / "validation_predictions.csv"
)

mlp = pd.read_csv(mlp_path)

# Step 13 saved predictions in the same
# order as validation.csv.
if len(mlp) != len(val):
    raise ValueError(
        "MLP prediction count does not "
        "match validation data."
    )

val_with_mlp = val.copy()
val_with_mlp["mlp_prediction"] = (
    mlp["predicted"].to_numpy()
)

mlp_lookup = val_with_mlp.set_index(
    "route"
)["mlp_prediction"]

mlp_aligned = eligible["route"].map(
    mlp_lookup
)

if mlp_aligned.isna().any():
    raise ValueError("Missing MLP predictions.")

scores.append(
    evaluate(
        "MLP",
        actual,
        mlp_aligned.to_numpy()
    )
)


# ---------------------------------------
# 7. Add Ridge and XGBoost predictions
# ---------------------------------------
additional_models = {
    "Ridge": (
        RESULTS
        / "ridge_overfitting"
        / "ridge_validation_predictions.csv"
    ),
    "XGBoost": (
        RESULTS
        / "tree_feature_engineering"
        / "xgboost_validation_predictions.csv"
    )
}

for model_name, path in additional_models.items():

    predictions = pd.read_csv(path)

    # Both files should contain one prediction
    # for each December validation record.
    if len(predictions) != len(val):
        raise ValueError(
            f"{model_name}: prediction count "
            "does not match validation data."
        )

    required_columns = {
        "route", "actual", "predicted"
    }

    if not required_columns.issubset(
        predictions.columns
    ):
        raise ValueError(
            f"{model_name}: missing columns."
        )

    # Check that predictions match the
    # original December validation records.
    if not np.array_equal(
        predictions["route"].to_numpy(),
        val["route"].to_numpy()
    ):
        raise ValueError(
            f"{model_name}: route order mismatch."
        )

    if not np.allclose(
        predictions["actual"].to_numpy(),
        val["median_eur"].to_numpy()
    ):
        raise ValueError(
            f"{model_name}: actual price mismatch."
        )

    # Predictions were saved in validation order.
    # Attach them before filtering eligible routes.
    aligned_df = val.copy()

    aligned_df["model_prediction"] = (
        predictions["predicted"].to_numpy()
    )

    lookup = aligned_df.set_index(
        "route"
    )["model_prediction"]

    aligned = eligible["route"].map(
        lookup
    )

    if aligned.isna().any():
        raise ValueError(
            f"{model_name}: missing predictions."
        )

    scores.append(
        evaluate(
            model_name,
            actual,
            aligned.to_numpy()
        )
    )

# ---------------------------------------
# 8. Final comparison
# ---------------------------------------
comparison = pd.DataFrame(scores)

comparison = comparison.sort_values(
    "MAE"
).reset_index(drop=True)

comparison.to_csv(
    OUT / "fair_comparison.csv",
    index=False
)

print("\n===== COMPLETE MODEL COMPARISON =====")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ---------------------------------------
# 9. Plot MAE
# ---------------------------------------
plt.figure(figsize=(11, 5))

plt.bar(
    comparison["Model"],
    comparison["MAE"]
)

plt.ylabel("December Validation MAE (EUR)")
plt.title(
    "Model Comparison on the Same Flight Routes"
)

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(
    OUT / "fair_comparison.png",
    dpi=200
)

plt.close()

# ---------------------------------------
# 10. Plot RMSE
# ---------------------------------------
plt.figure(figsize=(11, 5))

plt.bar(
    comparison["Model"],
    comparison["RMSE"]
)

plt.ylabel("December Validation RMSE (EUR)")
plt.title(
    "Model RMSE Comparison on the Same Flight Routes"
)

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(
    OUT / "rmse_comparison.png",
    dpi=200
)

plt.close()

print("\nResults saved to:", OUT)
print("Step 15 completed successfully!")
