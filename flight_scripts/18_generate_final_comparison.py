
import ssl
import certifi
from pathlib import Path
from io import StringIO
import urllib.request
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================================================
# PATHS
# =========================================================
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
OUT = RESULTS / "parent_paper_comparison"
OUT.mkdir(parents=True, exist_ok=True)

# =========================================================
# HISTORICAL EXCHANGE RATE (ECB)
# =========================================================
RATE_DATE = "2024-11-25"

url = (
    "https://data-api.ecb.europa.eu/service/data/"
    "EXR/D.USD.EUR.SP00.A"
    f"?startPeriod={RATE_DATE}&endPeriod={RATE_DATE}"
    "&format=csvdata"
)

print("Retrieving exchange rate from ECB...")

req = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0"}
)

ssl_context = ssl.create_default_context(
    cafile=certifi.where()
)

with urllib.request.urlopen(
    req,
    timeout=30,
    context=ssl_context
) as response:
    csv_data = response.read().decode("utf-8")

rate_df = pd.read_csv(StringIO(csv_data))

if rate_df.empty or "OBS_VALUE" not in rate_df.columns:
    raise ValueError("ECB did not return a valid exchange rate.")

usd_per_eur = float(rate_df["OBS_VALUE"].iloc[0])
usd_to_eur = 1 / usd_per_eur

print(f"1 USD = {usd_to_eur:.6f} EUR")

pd.DataFrame([{
    "date": RATE_DATE,
    "usd_per_eur": usd_per_eur,
    "eur_per_usd": usd_to_eur,
    "source": url
}]).to_csv(OUT / "FINAL_exchange_rate.csv", index=False)

# =========================================================
# PARENT PAPER RESULTS
# Average values reported in Table II
# =========================================================
parent = {
    "Linear Regression": (57.3, 5472.0),
    "Decision Tree": (52.0, 6137.0),
    "KNN": (65.7, 7146.6),
    "RNN": (52.7, 4729.8),
    "Bi-RNN": (52.1, 4671.8),
    "GRU": (50.1, 4260.9),
    "Bi-GRU": (47.8, 3957.0),
    "LSTM": (46.9, 3869.2),
    "Bi-LSTM": (46.7, 3852.6),
}

# =========================================================
# LOAD OUR EXISTING FAIR COMPARISON
# =========================================================
fair_path = (
    RESULTS / "fair_model_comparison" / "fair_comparison.csv"
)
fair = pd.read_csv(fair_path)

required = {"Model", "MAE", "RMSE"}
if not required.issubset(fair.columns):
    raise ValueError(
        f"Expected {required}, found {fair.columns.tolist()}"
    )

ours = {}
for _, row in fair.iterrows():
    ours[str(row["Model"]).strip()] = {
        "MAE": float(row["MAE"]),
        "RMSE": float(row["RMSE"])
    }

# =========================================================
# LOAD DECEMBER VALIDATION DATA AND IDENTIFY SHARED SUBSET
# =========================================================
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")

for df in (train, val):
    df["route"] = (
        df["origin_city"].astype(str)
        + " -> "
        + df["destination_city"].astype(str)
    )

oct_routes = set(
    train.loc[
        (train["year"] == 2026) &
        (train["month"] == 10),
        "route"
    ]
)
nov_routes = set(
    train.loc[
        (train["year"] == 2026) &
        (train["month"] == 11),
        "route"
    ]
)

mask = (
    (val["year"] == 2026) &
    (val["month"] == 12) &
    val["route"].isin(oct_routes & nov_routes)
).to_numpy()

print("Shared evaluation records:", mask.sum())

if mask.sum() != 31058:
    raise ValueError(
        "Shared subset is not 31,058 rows. "
        "Check the route eligibility logic."
    )

# =========================================================
# ADD KNN AND DECISION TREE ON THE SHARED SUBSET
# =========================================================
prediction_files = {
    "KNN": "knn_validation_predictions.csv",
    "Decision Tree": "decision_tree_validation_predictions.csv"
}

for model, filename in prediction_files.items():
    df = pd.read_csv(OUT / filename)

    if len(df) != len(val):
        raise ValueError(
            f"{model}: prediction count does not match validation."
        )

    if not np.array_equal(
        df["route"].astype(str).to_numpy(),
        val["route"].astype(str).to_numpy()
    ):
        raise ValueError(f"{model}: route order mismatch.")

    if not np.allclose(
        df["actual"].to_numpy(),
        val["median_eur"].to_numpy()
    ):
        raise ValueError(f"{model}: actual fare order mismatch.")

    actual = df.loc[mask, "actual"].to_numpy()
    predicted = df.loc[mask, "predicted"].to_numpy()

    ours[model] = {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        )
    }

# =========================================================
# BUILD COMPLETE MODEL TABLE
# =========================================================
model_order = [
    "Linear Regression",
    "Decision Tree",
    "KNN",
    "Ridge",
    "XGBoost",
    "MLP",
    "RNN",
    "Bi-RNN",
    "GRU",
    "Bi-GRU",
    "LSTM",
    "Bi-LSTM"
]

rows = []

for model in model_order:
    parent_mae, parent_mse = parent.get(
        model, (None, None)
    )
    our_metrics = ours.get(model)

    rows.append({
        "Model": model,
        "Parent_MAE_EUR": (
            parent_mae * usd_to_eur
            if parent_mae is not None else np.nan
        ),
        "Parent_RMSE_EUR": (
            np.sqrt(parent_mse) * usd_to_eur
            if parent_mse is not None else np.nan
        ),
        "Our_MAE_EUR": (
            our_metrics["MAE"]
            if our_metrics else np.nan
        ),
        "Our_RMSE_EUR": (
            our_metrics["RMSE"]
            if our_metrics else np.nan
        )
    })

comparison = pd.DataFrame(rows)

comparison.to_csv(
    OUT / "FINAL_model_comparison.csv",
    index=False
)

# =========================================================
# GENERATE LATEX TABLE
# =========================================================
def fmt(value):
    return "--" if pd.isna(value) else f"{value:.2f}"

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Comparison of airfare prediction models "
    r"reported in the parent study and our experiments.}",
    r"\label{tab:final_parent_comparison}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"& \multicolumn{2}{c}{Parent Study (EUR)} "
    r"& \multicolumn{2}{c}{Our Study (EUR)} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r"Model & MAE & RMSE & MAE & RMSE \\",
    r"\midrule"
]

for _, row in comparison.iterrows():
    lines.append(
        f"{row['Model']} & "
        f"{fmt(row['Parent_MAE_EUR'])} & "
        f"{fmt(row['Parent_RMSE_EUR'])} & "
        f"{fmt(row['Our_MAE_EUR'])} & "
        f"{fmt(row['Our_RMSE_EUR'])} "
        + r"\\"
    )

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\vspace{1mm}",
    r"\begin{minipage}{\columnwidth}",
    r"\footnotesize",
    r"\textit{Note:} All error values are in EUR. "
    r"Parent results are averages reported in "
    r"\cite{meepaganithage_airfare} and converted "
    r"from USD using the ECB rate on "
    + RATE_DATE
    + f" (1 USD = {usd_to_eur:.6f} EUR). "
    + r"Parent RMSE is computed as the square root "
    r"of reported average MSE. Our results use "
    r"31,058 December 2026 validation records. "
    r"The studies use different datasets and "
    r"prediction targets; values are not directly "
    r"comparable.",
    r"\end{minipage}",
    r"\end{table}"
]

latex = "\n".join(lines)

latex_file = OUT / "FINAL_comparison_table.tex"
latex_file.write_text(latex, encoding="utf-8")

print("\nFINAL COMPARISON:")
print(comparison.round(2).to_string(index=False))
print("\nSaved:", latex_file)
print("DONE.")
