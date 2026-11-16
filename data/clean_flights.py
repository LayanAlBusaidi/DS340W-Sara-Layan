from pathlib import Path
import pandas as pd

# File paths
DATA_DIR = Path(__file__).resolve().parent

INPUT_FILE = DATA_DIR / "fares-by-month.csv"
OUTPUT_FILE = DATA_DIR / "fares_by_month_cleaned.csv"

# 2. Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original dataset:", df.shape)
print("Missing values before cleaning:")
print(df.isnull().sum())

# 3. Remove extra spaces from city names
for col in ["origin_city", "destination_city"]:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

# 4. Standardize departure months
df["departure_month"] = pd.to_datetime(
    df["departure_month"],
    format="%Y-%m",
    errors="coerce"
)

# 5. Convert numerical columns
numeric_columns = [
    "observations",
    "median_eur",
    "p10_eur",
    "min_eur"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 6. Remove missing or invalid records
before = len(df)

df = df.dropna(subset=[
    "origin_city",
    "destination_city",
    "departure_month",
    *numeric_columns
])

df = df[
    (df["origin_city"] != "") &
    (df["destination_city"] != "") &
    (df["origin_city"] != df["destination_city"]) &
    (df["observations"] > 0) &
    (df["min_eur"] > 0) &
    (df["min_eur"] <= df["p10_eur"]) &
    (df["p10_eur"] <= df["median_eur"])
].copy()

print("Invalid/missing rows removed:", before - len(df))

# 7. Remove exact duplicate records
before = len(df)
df = df.drop_duplicates()
print("Exact duplicates removed:", before - len(df))

# 8. Check for conflicting route-month records
keys = ["origin_city", "destination_city", "departure_month"]

conflicts = df.duplicated(subset=keys, keep=False)

if conflicts.any():
    print(df.loc[conflicts, keys + numeric_columns].head(10))
    raise ValueError(
        "Conflicting records exist for the same route and month. "
        "Review these records before continuing."
    )

# 9. Restore YYYY-MM date format
df["departure_month"] = df["departure_month"].dt.strftime("%Y-%m")

# 10. Sort records
df = df.sort_values(
    by=["origin_city", "destination_city", "departure_month"]
).reset_index(drop=True)

# 11. Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

# 12. Display summary
print("\nCLEANING COMPLETE")
print("----------------------------")
print("Final rows:", len(df))
print("Final columns:", len(df.columns))
print("Unique origin cities:", df["origin_city"].nunique())
print("Unique destination cities:", df["destination_city"].nunique())
print("Earliest departure month:", df["departure_month"].min())
print("Latest departure month:", df["departure_month"].max())
print("Remaining missing values:", df.isnull().sum().sum())
print("Saved to:", OUTPUT_FILE)
print("Python script executed successfully!")