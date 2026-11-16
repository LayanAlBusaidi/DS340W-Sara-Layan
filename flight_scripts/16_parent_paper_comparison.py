
from pathlib import Path
import pandas as pd

# ---------------------------------------
# 1. File paths
# ---------------------------------------
ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / "results"
OUT = RESULTS / "parent_paper_comparison"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 2. Parent paper results
# Table II: Average MAE
# Units are not confirmed as EUR.
# ---------------------------------------
parent_results = {
    "Linear Regression": 57.3,
    "Decision Tree": 52.0,
    "KNN": 65.7,
    "RNN": 52.7,
    "Bi-RNN": 52.1,
    "GRU": 50.1,
    "Bi-GRU": 47.8,
    "LSTM": 46.9,
    "Bi-LSTM": 46.7
}

# ---------------------------------------
# 3. Load our experimental results
# All our airfare errors are in EUR.
# ---------------------------------------
our_results_path = (
    RESULTS
    / "fair_model_comparison"
    / "fair_comparison.csv"
)

our_df = pd.read_csv(our_results_path)

# Keep only our six machine learning models.
selected_models = [
    "Ridge",
    "XGBoost",
    "MLP",
    "RNN",
    "GRU",
    "LSTM"
]

our_df = our_df[
    our_df["Model"].isin(selected_models)
].copy()

our_results = dict(
    zip(our_df["Model"], our_df["MAE"])
)

# ---------------------------------------
# 4. Combine results
# ---------------------------------------
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
    rows.append({
        "Model": model,
        "Parent_MAE_Original_Units": (
            parent_results.get(model)
        ),
        "Our_MAE_EUR": (
            our_results.get(model)
        )
    })

comparison = pd.DataFrame(rows)

comparison.to_csv(
    OUT / "parent_vs_our_models.csv",
    index=False
)

print("\n===== PARENT VS OUR MODELS =====")
print(comparison.to_string(index=False))

# ---------------------------------------
# 5. Generate LaTeX table for Overleaf
# ---------------------------------------
def format_value(value):
    if pd.isna(value):
        return "--"
    return f"{value:.2f}"

latex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Comparison of airfare prediction models "
    r"reported in the parent study and our experiments.}",
    r"\label{tab:parent_comparison}",
    r"\small",
    r"\begin{tabular}{lcc}",
    r"\toprule",
    r"Model & Parent MAE & Our MAE (EUR) \\",
    r"\midrule"
]

for _, row in comparison.iterrows():

    model = row["Model"]
    parent_mae = format_value(
        row["Parent_MAE_Original_Units"]
    )
    our_mae = format_value(
        row["Our_MAE_EUR"]
    )

    latex_lines.append(
        f"{model} & {parent_mae} & {our_mae} "
        + r"\\"
    )

latex_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\vspace{1mm}",
    r"\begin{minipage}{\columnwidth}",
    r"\footnotesize",
    r"\textit{Note:} Parent MAE values are averages "
    r"reported in \cite{meepaganithage_airfare} "
    r"in unspecified price units. Our MAE values "
    r"are in EUR and use 31,058 December validation "
    r"records. The datasets and prediction targets "
    r"differ, so the numerical errors are not "
    r"directly comparable.",
    r"\end{minipage}",
    r"\end{table}"
])

latex_text = "\n".join(latex_lines)

latex_path = OUT / "parent_comparison_table.tex"

latex_path.write_text(
    latex_text,
    encoding="utf-8"
)

print("\n===== LATEX TABLE =====")
print(latex_text)

print("\nFiles saved to:", OUT)

