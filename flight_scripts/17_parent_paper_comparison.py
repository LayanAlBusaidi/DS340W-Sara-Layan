
from pathlib import Path
from io import StringIO
import urllib.request
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# 1. Paths
# ==================================================
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
OUT = RESULTS / "parent_paper_comparison"

OUT.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. Retrieve historical exchange rate from ECB
# ==================================================
RATE_DATE = "2024-11-25"
RATE_SOURCE = "European Central Bank"

url = (
    "https://data-api.ecb.europa.eu/service/data/"
    "EXR/D.USD.EUR.SP00.A"
    f"?startPeriod={RATE_DATE}"
    f"&endPeriod={RATE_DATE}"
    "&format=csvdata"
)

print("\nRetrieving historical exchange rate...")

request = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0"}
)

with urllib.request.urlopen(
    request, timeout=30
) as response:
    csv_text = response.read().decode("utf-8")

rate_df = pd.read_csv(StringIO(csv_text))

if rate_df.empty:
    raise ValueError(
        "No exchange rate returned for the selected date."
    )

# ECB series D.USD.EUR.SP00.A quotes USD per 1 EUR.
usd_per_eur = float(rate_df["OBS_VALUE"].iloc[0])
USD_TO_EUR = 1.0 / usd_per_eur

print(f"Date: {RATE_DATE}")
print(f"1 EUR = {usd_per_eur:.6f} USD")
print(f"1 USD = {USD_TO_EUR:.6f} EUR")

# Save conversion information.
pd.DataFrame([{
    "Date": RATE_DATE,
    "Source": RATE_SOURCE,
    "USD_per_EUR": usd_per_eur,
    "EUR_per_USD": USD_TO_EUR,
    "URL": url
}]).to_csv(
    OUT / "exchange_rate.csv",
    index=False
)

# ==================================================
# 3. Parent paper results
# Table II: Average across four window sizes
# ==================================================
parent_results = {
    "Linear Regression": {
        "MAE": 57.3,
        "MSE": 5472.0
    },
    "Decision Tree": {
        "MAE": 52.0,
        "MSE": 6137.0
    },
    "KNN": {
        "MAE": 65.7,
        "MSE": 7146.6
    },
    "RNN": {
        "MAE": 52.7,
        "MSE": 4729.8
    },
    "GRU": {
        "MAE": 50.1,
        "MSE": 4260.9
    },
    "LSTM": {
        "MAE": 46.9,
        "MSE": 3869.2
    }
}

# ==================================================
# 4. Load our training and validation datasets
# ==================================================
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")

for frame in (train, val):
    frame["route"] = (
        frame["origin_city"].astype(str)
        + " -> "
        + frame["destination_city"].astype(str)
    )

# ==================================================
# 5. Identify the common December subset
# ==================================================
# Eligible routes have fare history in October
# and November, matching our recurrent evaluation.

oct_routes = set(
    train.loc[
        (train["year"] == 2026)
        & (train["month"] == 10),
        "route"
    ]
)

nov_routes = set(
    train.loc[
        (train["year"] == 2026)
        & (train["month"] == 11),
        "route"
    ]
)

common_routes = oct_routes & nov_routes

eligible_mask = (
    (val["year"] == 2026)
    & (val["month"] == 12)
    & val["route"].isin(common_routes)
)

eligible_mask = eligible_mask.to_numpy()

n_eligible = int(eligible_mask.sum())

print("\nCommon December validation records:", n_eligible)

if n_eligible != 31058:
    raise ValueError(
        f"Expected 31,058 records but found "
        f"{n_eligible}. Check the subset selection "
        "against Step 15 before comparing results."
    )

# ==================================================
# 6. Load our existing fair model results
# ==================================================
fair = pd.read_csv(
    RESULTS
    / "fair_model_comparison"
    / "fair_comparison.csv"
)

print("\nExisting comparison columns:")
print(fair.columns.tolist())

required = {"Model", "MAE", "RMSE", "R2"}

if not required.issubset(fair.columns):
    raise ValueError(
        "Missing required columns in "
        "fair_comparison.csv"
    )

our_results = {}

for _, row in fair.iterrows():
    our_results[str(row["Model"])] = {
        "MAE": float(row["MAE"]),
        "RMSE": float(row["RMSE"]),
        "R2": float(row["R2"])
    }

# ==================================================
# 7. Calculate KNN and Decision Tree
# on the same December records
# ==================================================
files = {
    "KNN": "knn_validation_predictions.csv",
    "Decision Tree":
        "decision_tree_validation_predictions.csv"
}

for model_name, filename in files.items():

    df = pd.read_csv(OUT / filename)

    if len(df) != len(val):
        raise ValueError(
            f"{model_name}: prediction row count "
            "does not match validation dataset."
        )

    expected_actual = val["median_eur"].to_numpy()

    if not np.allclose(
        df["actual"].to_numpy(),
        expected_actual
    ):
        raise ValueError(
            f"{model_name}: predictions are not "
            "aligned with validation records."
        )

    if not np.array_equal(
        df["route"].to_numpy(),
        val["route"].to_numpy()
    ):
        raise ValueError(
            f"{model_name}: route order mismatch."
        )

    actual = df.loc[
        eligible_mask, "actual"
    ].to_numpy()

    predicted = df.loc[
        eligible_mask, "predicted"
    ].to_numpy()

    our_results[model_name] = {
        "MAE": mean_absolute_error(
            actual, predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }

# The parent uses ordinary linear regression.
# Our closest equivalent is route-aware Ridge.
our_results["Linear Regression"] = our_results["Ridge"]

# ==================================================
# 8. Build comparison table
# ==================================================
shared_models = [
    "Linear Regression",
    "Decision Tree",
    "KNN",
    "RNN",
    "GRU",
    "LSTM"
]

rows = []

for model in shared_models:

    parent_mae_usd = parent_results[model]["MAE"]
    parent_mse_usd = parent_results[model]["MSE"]

    # MAE has units of currency.
    parent_mae_eur = parent_mae_usd * USD_TO_EUR

    # MSE has squared currency units.
    # Convert to RMSE first, then to EUR.
    parent_rmse_eur = (
        np.sqrt(parent_mse_usd) * USD_TO_EUR
    )

    ours = our_results[model]

    rows.append({
        "Model": model,
        "Parent_MAE_USD": parent_mae_usd,
        "Parent_MAE_EUR": parent_mae_eur,
        "Parent_RMSE_EUR": parent_rmse_eur,
        "Our_MAE_EUR": ours["MAE"],
        "Our_RMSE_EUR": ours["RMSE"],
        "Our_R2": ours["R2"]
    })

comparison = pd.DataFrame(rows)

comparison.to_csv(
    OUT / "shared_models_comparison.csv",
    index=False
)

print("\n===== PARENT VS OUR RESULTS =====")

print(
    comparison[
        [
            "Model",
            "Parent_MAE_EUR",
            "Our_MAE_EUR",
            "Parent_RMSE_EUR",
            "Our_RMSE_EUR"
        ]
    ].round(2).to_string(index=False)
)

# ==================================================
# 9. Generate LaTeX table for Overleaf
# ==================================================
lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Comparison of shared airfare forecasting "
    r"models between the parent study and our approach.}",
    r"\label{tab:parent_comparison}",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"& \multicolumn{2}{c}{Parent Study} "
    r"& \multicolumn{2}{c}{Our Study} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r"Model & MAE & RMSE & MAE & RMSE \\",
    r"\midrule"
]

for _, row in comparison.iterrows():

    model = row["Model"]

    if model == "Linear Regression":
        model = "Linear / Ridge"

    lines.append(
        f"{model} & "
        f"{row['Parent_MAE_EUR']:.2f} & "
        f"{row['Parent_RMSE_EUR']:.2f} & "
        f"{row['Our_MAE_EUR']:.2f} & "
        f"{row['Our_RMSE_EUR']:.2f} "
        + r"\\"
    )

lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\vspace{1mm}",
    r"\begin{minipage}{\columnwidth}",
    r"\footnotesize",
    r"\textit{Note:} All errors are expressed in EUR. "
    r"Parent results are averages from "
    r"\cite{meepaganithage_airfare}, converted from "
    r"USD using the ECB reference rate of "
    + f"1 USD = {USD_TO_EUR:.6f} EUR "
    + f"({RATE_DATE}). "
    + r"Parent RMSE is calculated from the "
    r"reported average MSE. Our results use "
    r"31,058 December 2026 validation records. "
    r"Our linear model uses Ridge regularization. "
    r"Different datasets and targets prevent "
    r"direct numerical performance comparisons.",
    r"\end{minipage}",
    r"\end{table}"
])

latex_text = "\n".join(lines)

latex_path = OUT / "parent_comparison_table.tex"

latex_path.write_text(
    latex_text,
    encoding="utf-8"
)

# ==================================================
# 10. Save our additional models separately
# ==================================================
additional_models = [
    "Ridge",
    "XGBoost",
    "MLP"
]

extra_rows = []

for model in additional_models:
    if model in our_results:
        extra_rows.append({
            "Model": model,
            **our_results[model]
        })

pd.DataFrame(extra_rows).to_csv(
    OUT / "additional_models.csv",
    index=False
)

print("\n===== FILES GENERATED =====")
print(OUT / "exchange_rate.csv")
print(OUT / "shared_models_comparison.csv")
print(OUT / "parent_comparison_table.tex")
print(OUT / "additional_models.csv")
print("\nComplete!")
