
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. Paths
# ----------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "deep_learning_data_check"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------
# 2. Load data
# ----------------------------------------
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")

df = pd.concat(
    [train, val],
    ignore_index=True
)

df["departure_month"] = pd.to_datetime(
    df["departure_month"]
)

df["route"] = (
    df["origin_city"].astype(str)
    + " -> "
    + df["destination_city"].astype(str)
)

print("\n===== DATASET OVERVIEW =====")
print("Total records:", len(df))
print("Unique routes:", df["route"].nunique())

# ----------------------------------------
# 3. Check repeated route observations
# ----------------------------------------
route_months = (
    df.groupby("route")["departure_month"]
    .nunique()
)

print("\n===== MONTHS PER ROUTE =====")
print(route_months.describe())

for n in [1, 2, 3, 4]:
    count = (route_months >= n).sum()
    percentage = 100 * count / len(route_months)

    print(
        f"Routes with >= {n} months: "
        f"{count:,} ({percentage:.2f}%)"
    )

# ----------------------------------------
# 4. Check duplicate route-month records
# ----------------------------------------
duplicates = df.duplicated(
    subset=["route", "departure_month"],
    keep=False
)

print("\n===== DUPLICATE ROUTE-MONTH RECORDS =====")
print("Rows belonging to duplicates:", duplicates.sum())

if duplicates.any():
    print(
        df.loc[
            duplicates,
            ["route", "departure_month", "median_eur"]
        ].head(10).to_string(index=False)
    )

# ----------------------------------------
# 5. Check month coverage
# ----------------------------------------
print("\n===== MONTH COVERAGE =====")

coverage = (
    df.groupby("departure_month")
    .agg(
        rows=("median_eur", "size"),
        unique_routes=("route", "nunique")
    )
)

print(coverage.to_string())

# ----------------------------------------
# 6. Check sequence availability
# ----------------------------------------
# For a genuine 2-month-history forecast,
# a route needs two earlier consecutive months
# before the prediction month.

route_month_pairs = set(
    zip(
        df["route"],
        df["departure_month"].dt.to_period("M")
    )
)

print("\n===== SEQUENCE AVAILABILITY =====")

for target_month in sorted(
    df["departure_month"].dt.to_period("M").unique()
):

    target_rows = df[
        df["departure_month"].dt.to_period("M")
        == target_month
    ]

    previous_1 = target_month - 1
    previous_2 = target_month - 2

    valid = target_rows["route"].apply(
        lambda route: (
            (route, previous_1) in route_month_pairs
            and
            (route, previous_2) in route_month_pairs
        )
    )

    print(
        f"{target_month}: "
        f"{valid.sum():,} / {len(target_rows):,} "
        f"rows have 2 prior consecutive months"
    )

# ----------------------------------------
# 7. Plot route history distribution
# ----------------------------------------
distribution = (
    route_months.value_counts()
    .sort_index()
)

plt.figure(figsize=(8, 5))

plt.bar(
    distribution.index.astype(str),
    distribution.values
)

plt.xlabel("Number of distinct months per route")
plt.ylabel("Number of routes")
plt.title("Historical Month Coverage by Flight Route")
plt.tight_layout()

plt.savefig(
    OUT / "route_month_distribution.png",
    dpi=200
)

plt.close()

# ----------------------------------------
# 8. Save results
# ----------------------------------------
route_months.rename(
    "months_available"
).to_csv(
    OUT / "route_month_counts.csv"
)

coverage.to_csv(
    OUT / "monthly_coverage.csv"
)

print("\nResults saved to:", OUT)
