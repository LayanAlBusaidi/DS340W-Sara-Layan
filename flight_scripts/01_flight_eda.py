
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# 1. File paths
# -----------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "fares_by_month_cleaned.csv"
OUTPUT_DIR = ROOT / "results" / "flight_eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. Load dataset
# -----------------------------------
df = pd.read_csv(DATA_FILE)

print("\n===== DATASET OVERVIEW =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nUnique origin cities:", df["origin_city"].nunique())
print("Unique destination cities:", df["destination_city"].nunique())

# Create a unique route identifier
df["route"] = (
    df["origin_city"] + " -> " + df["destination_city"]
)

print("Unique routes:", df["route"].nunique())

# -----------------------------------
# 3. Departure month coverage
# -----------------------------------
monthly = (
    df.groupby("departure_month")
    .agg(
        routes=("route", "nunique"),
        fare_records=("median_eur", "size"),
        fare_observations=("observations", "sum"),
        median_of_route_fares=("median_eur", "median")
    )
    .reset_index()
    .sort_values("departure_month")
)

print("\n===== MONTHLY COVERAGE =====")
print(monthly.to_string(index=False))

monthly.to_csv(
    OUTPUT_DIR / "monthly_summary.csv",
    index=False
)

# -----------------------------------
# 4. Fare price distribution
# -----------------------------------
print("\n===== AIRFARE STATISTICS =====")
print(df["median_eur"].describe(
    percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
))

plt.figure(figsize=(10, 5))
plt.hist(df["median_eur"], bins=80)
plt.title("Distribution of Monthly Median Airfares")
plt.xlabel("Monthly median airfare (EUR)")
plt.ylabel("Number of route-month records")
plt.xlim(0, df["median_eur"].quantile(0.99))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fare_distribution.png", dpi=200)
plt.close()

# -----------------------------------
# 5. Median airfare by month
# -----------------------------------
plt.figure(figsize=(10, 5))
plt.plot(
    monthly["departure_month"],
    monthly["median_of_route_fares"],
    marker="o"
)
plt.title("Median Route Airfare by Departure Month")
plt.xlabel("Departure month")
plt.ylabel("Median airfare (EUR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_fares.png", dpi=200)
plt.close()

# -----------------------------------
# 6. Route history and sequence lengths
# -----------------------------------
route_month_counts = (
    df.groupby("route")["departure_month"]
    .nunique()
)

print("\n===== ROUTE MONTH COVERAGE =====")
print(route_month_counts.value_counts().sort_index())

for minimum in [2, 3, 4, 5, 6]:
    eligible = (route_month_counts >= minimum).sum()
    print(f"Routes with at least {minimum} months: {eligible}")

route_month_counts.rename("months_available").to_csv(
    OUTPUT_DIR / "route_month_coverage.csv"
)

plt.figure(figsize=(8, 5))
route_month_counts.value_counts().sort_index().plot(kind="bar")
plt.title("Number of Departure Months Available per Route")
plt.xlabel("Number of distinct departure months")
plt.ylabel("Number of routes")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "route_month_coverage.png", dpi=200)
plt.close()

# -----------------------------------
# 7. Consecutive month coverage
# -----------------------------------
df["month_date"] = pd.to_datetime(
    df["departure_month"], format="%Y-%m"
)

df["month_index"] = (
    df["month_date"].dt.year * 12
    + df["month_date"].dt.month
)

def longest_consecutive_run(months):
    values = sorted(set(months))
    if not values:
        return 0

    longest = 1
    current = 1

    for previous, following in zip(values, values[1:]):
        if following - previous == 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)

    return longest

consecutive_runs = (
    df.groupby("route")["month_index"]
    .agg(longest_consecutive_run)
)

print("\n===== CONSECUTIVE MONTH COVERAGE =====")
print(consecutive_runs.value_counts().sort_index())

for minimum in [3, 4, 5, 6]:
    eligible = (consecutive_runs >= minimum).sum()
    print(
        f"Routes with {minimum}+ consecutive months: {eligible}"
    )

# -----------------------------------
# 8. Save overall summary
# -----------------------------------
summary = pd.DataFrame([{
    "total_records": len(df),
    "unique_routes": df["route"].nunique(),
    "origin_cities": df["origin_city"].nunique(),
    "destination_cities": df["destination_city"].nunique(),
    "earliest_month": df["departure_month"].min(),
    "latest_month": df["departure_month"].max(),
    "median_fare_eur": df["median_eur"].median(),
    "routes_3plus_months": int(
        (route_month_counts >= 3).sum()
    ),
    "routes_3plus_consecutive_months": int(
        (consecutive_runs >= 3).sum()
    )
}])

summary.to_csv(OUTPUT_DIR / "eda_summary.csv", index=False)

print("\n===== EDA COMPLETE =====")
print("Results saved to:", OUTPUT_DIR)
