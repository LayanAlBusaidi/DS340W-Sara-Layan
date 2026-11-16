
from pathlib import Path
import pandas as pd
import unicodedata

# 1. Set file paths
ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT / "data" / "daily_cost" / "raw" / "daily_cost.csv"
OUTPUT_DIR = ROOT / "data" / "daily_cost" / "clean"
OUTPUT_FILE = OUTPUT_DIR / "clean_daily_cost.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load dataset and select relevant features
df = pd.read_csv(RAW_FILE)
df.columns = df.columns.str.strip()

print(f"Original rows: {len(df)}")

columns = {
    "city": "city",
    "country": "country",
    "x1": "meal_inexpensive_usd",
    "x2": "meal_midrange_two_usd",
    "x3": "fast_food_usd",
    "x6": "coffee_usd",
    "x7": "soft_drink_usd",
    "x8": "water_usd",
    "x28": "transport_ticket_usd",
    "x29": "transport_monthly_usd",
    "x30": "taxi_start_usd",
    "x31": "taxi_per_km_usd",
    "x41": "cinema_usd",
    "data_quality": "data_quality"
}

df = df[list(columns.keys())].rename(columns=columns)

# 3. Standardize city and country names
def normalize_name(value):
    if pd.isna(value):
        return None

    value = unicodedata.normalize("NFKC", str(value))
    return " ".join(value.strip().lower().split())

df["city"] = df["city"].apply(normalize_name)
df["country"] = df["country"].apply(normalize_name)

df = df.dropna(subset=["city", "country"])
df = df[(df["city"] != "") & (df["country"] != "")]

# 4. Convert prices to numeric values
price_columns = [
    column for column in df.columns
    if column not in ["city", "country", "data_quality"]
]

for column in price_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df.loc[df[column] < 0, column] = float("nan")

df["data_quality"] = pd.to_numeric(
    df["data_quality"], errors="coerce"
)

# 5. Remove duplicate cities and unusable rows
df["missing_count"] = df[price_columns].isna().sum(axis=1)

df = df.sort_values(
    ["missing_count", "data_quality"],
    ascending=[True, False]
)

df = df.drop_duplicates(subset=["city", "country"])
df = df.drop(columns=["missing_count"])
df = df.dropna(subset=price_columns, how="all")

# 6. Sort and save cleaned dataset
df = df.sort_values(["country", "city"]).reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Cleaned rows: {len(df)}")
print(f"Unique cities: {df['city'].nunique()}")
print(f"Unique countries: {df['country'].nunique()}")

print("\nMissing values:")
print(df.isna().sum())

print(f"\nSaved to: {OUTPUT_FILE}")

print("\nData quality distribution:")
print(df["data_quality"].value_counts(dropna=False))

# 7. Fill missing prices using country medians
estimated_df = df.copy()

for column in price_columns:
    missing_before = estimated_df[column].isna()

    country_median = estimated_df.groupby("country")[column].transform("median")

    estimated_df[column] = estimated_df[column].fillna(country_median)

    estimated_df[f"{column}_imputed"] = (
        missing_before & estimated_df[column].notna()
    ).astype(int)

estimated_df[price_columns] = estimated_df[price_columns].round(2)

# 8. Save estimated dataset
ESTIMATED_FILE = OUTPUT_DIR / "daily_cost_estimated.csv"

estimated_df.to_csv(ESTIMATED_FILE, index=False, float_format="%.2f")

print("\nMissing values after estimation:")
print(estimated_df[price_columns].isna().sum())

print("\nNumber of estimated values:")
for column in price_columns:
    print(f"{column}: {estimated_df[f'{column}_imputed'].sum()}")

print(f"\nSaved estimated dataset to: {ESTIMATED_FILE}")
