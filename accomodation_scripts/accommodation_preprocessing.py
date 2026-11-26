from pathlib import Path
import pandas as pd
import json
import requests
import time

# Required packages:
# pip install pandas requests

start_time = time.perf_counter()

# 1. Set file paths

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "accommodation" / "raw"
OUTPUT_DIR = ROOT / "data" / "accommodation" / "clean"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "combined_accommodation.csv"
rate_file = OUTPUT_DIR / "exchange_rates.csv"

# 2. Select relevant columns

SELECTED_COLUMNS = [
    "id",
    "host_id",
    "latitude",
    "longitude",
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "bathrooms_text",
    "amenities",
    "minimum_nights",
    "maximum_nights",
    "instant_bookable",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "reviews_per_month",
    "host_is_superhost",
    "last_scraped",
    "price_quote_checkin_date",
    "price_quote_checkout_date",
    "price_quote_total_price",
    "price_quote_price_per_night",
    "price_quote_raw",
    "price"
]

# 3. Extract information from price quote JSON

def extract_quote(value):
    try:
        quote = json.loads(value).get("quote", {})
        return quote if isinstance(quote, dict) else {}

    except (TypeError, ValueError, AttributeError):
        return {}


# 4. Load and combine accommodation datasets

print("\n--- LOADING ACCOMMODATION DATA ---")

files = sorted(RAW_DIR.rglob("listings*.csv"))

if not files:
    raise FileNotFoundError("No accommodation datasets found.")

datasets = []

print(f"Files found: {len(files)}")

for i, file in enumerate(files, start=1):

    try:
        df = pd.read_csv(
            file,
            usecols=lambda col: col in SELECTED_COLUMNS,
            dtype={"id": "string", "host_id": "string"},
            low_memory=False
        )

        df = df.reindex(columns=SELECTED_COLUMNS)

        # Preserve geographical and source information
        df["city"] = file.parent.name
        df["source_file"] = file.name

        # Extract pricing information from JSON
        quotes = df["price_quote_raw"].apply(extract_quote)

        df["currency"] = quotes.apply(
            lambda q: q.get("currency")
        )

        df["is_monthly_priced"] = quotes.apply(
            lambda q: q.get("is_monthly_priced")
        )

        df["quote_available"] = quotes.apply(
            lambda q: q.get("is_available")
        )

        # Remove raw JSON after extraction
        df = df.drop(columns=["price_quote_raw"])

        datasets.append(df)

        # Display progress every 10 files
        if i % 10 == 0 or i == len(files):
            print(f"Files processed: {i}/{len(files)}")

    except Exception as e:
        raise RuntimeError(
            f"Error processing {file}: {e}"
        ) from e

combined = pd.concat(datasets, ignore_index=True)

# Release memory occupied by individual datasets
del datasets, df, quotes

original_records = len(combined)

print(f"Original records: {original_records:,}")
print(f"Original locations: {combined['city'].nunique()}")

print(
    f"Loading time: "
    f"{time.perf_counter() - start_time:.2f} seconds"
)


# 5. Convert date columns

date_columns = [
    "last_scraped",
    "price_quote_checkin_date",
    "price_quote_checkout_date"
]

for column in date_columns:
    combined[column] = pd.to_datetime(
        combined[column],
        errors="coerce"
    )


# 6. Create temporal features

combined["booking_lead_days"] = (
    combined["price_quote_checkin_date"]
    - combined["last_scraped"]
).dt.days

combined["stay_duration"] = (
    combined["price_quote_checkout_date"]
    - combined["price_quote_checkin_date"]
).dt.days

combined["checkin_month"] = (
    combined["price_quote_checkin_date"].dt.month
)

combined["checkin_year"] = (
    combined["price_quote_checkin_date"].dt.year
)


# 7. Resolve missing currencies

print("\n--- CURRENCY RECOVERY ---")

# Infer currencies based on geographical location
# and previously analyzed price distributions.
# These currencies are inferred, not independently verified.

currency_mapping = {
    "bangkok": "THB",
    "istanbul": "TRY",
    "nairobi": "KES",
    "prague": "CZK"
}

# Exclude locations with uncertain currencies

excluded_cities = ["geneva", "vaud"]

before_removal = len(combined)

combined = combined[
    ~combined["city"].isin(excluded_cities)
].copy()

removed_locations = before_removal - len(combined)

print("Excluded locations:", ", ".join(excluded_cities))
print(f"Records removed: {removed_locations:,}")

print(
    f"Percentage removed: "
    f"{removed_locations / before_removal * 100:.2f}%"
)

# Preserve original currency information

combined["currency_resolved"] = combined["currency"]
combined["currency_imputed"] = 0

# Convert original nightly prices to numeric values

combined["price_quote_price_per_night"] = pd.to_numeric(
    combined["price_quote_price_per_night"],
    errors="coerce"
)

# Identify records with missing currencies and valid prices

mask = (
    combined["currency"].isna()
    & (combined["price_quote_price_per_night"] > 0)
    & (combined["price_quote_price_per_night"] < float("inf"))
    & combined["city"].isin(currency_mapping)
)

# Assign inferred currencies

combined.loc[mask, "currency_resolved"] = (
    combined.loc[mask, "city"].map(currency_mapping)
)

# Flag records with inferred currencies

combined.loc[mask, "currency_imputed"] = 1

print("\nRecovered currencies:")

print(
    combined.loc[mask, "currency_resolved"]
    .value_counts()
    .to_string()
)

print(f"Total recovered: {mask.sum():,}")

print(
    f"Unique currencies: "
    f"{combined['currency_resolved'].nunique()}"
)


# 8. Remove records without valid nightly prices

print("\n--- PRICE CLEANING ---")

before_cleaning = len(combined)

combined = combined[
    (combined["price_quote_price_per_night"] > 0)
    & (combined["price_quote_price_per_night"] < float("inf"))
    & combined["currency_resolved"].notna()
].copy()

print(f"Records removed: {before_cleaning - len(combined):,}")
print(f"Remaining records: {len(combined):,}")


# 9. Retrieve historical exchange rates

print("\n--- HISTORICAL EXCHANGE RATES ---")

currency_start = time.perf_counter()

# Use the collection date rather than the future check-in date

combined["scrape_date"] = (
    combined["last_scraped"].dt.strftime("%Y-%m-%d")
)

if combined["scrape_date"].isna().any():
    raise ValueError("Missing collection dates need investigation.")

rate_columns = [
    "scrape_date",
    "currency_resolved",
    "exchange_rate_to_eur",
    "exchange_rate_date"
]

# Load previously downloaded exchange rates

if rate_file.exists():

    cached_rates = pd.read_csv(rate_file)

    cached_rates["exchange_rate_to_eur"] = pd.to_numeric(
        cached_rates["exchange_rate_to_eur"],
        errors="coerce"
    )

    # Retain only valid cached rates

    cached_rates = cached_rates[
        (cached_rates["exchange_rate_to_eur"] > 0)
        & (cached_rates["exchange_rate_to_eur"] < float("inf"))
        & cached_rates["exchange_rate_date"].notna()
        & (
            cached_rates["exchange_rate_date"]
            <= cached_rates["scrape_date"]
        )
    ].copy()

else:
    cached_rates = pd.DataFrame(columns=rate_columns)

# Identify required currency and date combinations

required_pairs = combined[[
    "scrape_date",
    "currency_resolved"
]].drop_duplicates()

existing_pairs = set(
    zip(
        cached_rates["scrape_date"],
        cached_rates["currency_resolved"]
    )
)

new_rates = []

# Retrieve only exchange rates not already cached

for date, group in required_pairs.groupby("scrape_date"):

    currencies = [
        currency
        for currency in group["currency_resolved"]
        if (date, currency) not in existing_pairs
    ]

    if not currencies:
        continue

    # EUR does not require conversion

    if "EUR" in currencies:

        new_rates.append({
            "scrape_date": date,
            "currency_resolved": "EUR",
            "exchange_rate_to_eur": 1.0,
            "exchange_rate_date": date
        })

        currencies.remove("EUR")

    if not currencies:
        continue

    # Retrieve historical exchange rates from Frankfurter

    response = requests.get(
        "https://api.frankfurter.dev/v2/rates",
        params={
            "date": date,
            "base": "EUR",
            "quotes": ",".join(currencies)
        },
        timeout=60
    )

    response.raise_for_status()

    rates = response.json()

    for item in rates:

        currency = item["quote"]
        rate = float(item["rate"])

        if (
            currency in currencies
            and 0 < rate < float("inf")
            and item["date"] <= date
        ):

            new_rates.append({
                "scrape_date": date,
                "currency_resolved": currency,
                "exchange_rate_to_eur": 1 / rate,
                "exchange_rate_date": item["date"]
            })

# Combine cached and newly downloaded rates

new_rates_df = pd.DataFrame(
    new_rates,
    columns=rate_columns
)

exchange_rates = pd.concat(
    [cached_rates, new_rates_df],
    ignore_index=True
)

exchange_rates = exchange_rates.drop_duplicates(
    subset=["scrape_date", "currency_resolved"],
    keep="last"
)

# Ensure exchange rates have a numerical data type

exchange_rates["exchange_rate_to_eur"] = pd.to_numeric(
    exchange_rates["exchange_rate_to_eur"],
    errors="coerce"
)

# Save rates for reproducibility

exchange_rates.to_csv(rate_file, index=False)

print(f"Required currency-date pairs: {len(required_pairs):,}")
print(f"New exchange rates retrieved: {len(new_rates):,}")
print(f"Total cached exchange rates: {len(exchange_rates):,}")


# 10. Merge exchange rates with accommodation data

combined = combined.merge(
    exchange_rates,
    on=["scrape_date", "currency_resolved"],
    how="left",
    validate="many_to_one"
)

# Validate exchange rates before conversion

invalid_rate_mask = (
    combined["exchange_rate_to_eur"].isna()
    | (combined["exchange_rate_to_eur"] <= 0)
    | (combined["exchange_rate_to_eur"] == float("inf"))
)

print(f"Missing or invalid exchange rates: {invalid_rate_mask.sum():,}")

if invalid_rate_mask.any():

    print(
        combined.loc[invalid_rate_mask]
        .groupby([
            "currency_resolved",
            "scrape_date"
        ]).size()
    )

    raise ValueError(
        "Some exchange rates are missing or invalid."
    )


# 11. Convert accommodation prices to EUR

combined["nightly_price_eur"] = (
    combined["price_quote_price_per_night"]
    * combined["exchange_rate_to_eur"]
).round(2)

combined["price_quote_total_price"] = pd.to_numeric(
    combined["price_quote_total_price"],
    errors="coerce"
)

combined["total_price_eur"] = (
    combined["price_quote_total_price"]
    * combined["exchange_rate_to_eur"]
).round(2)

# Ensure converted prices are numerical

combined["nightly_price_eur"] = pd.to_numeric(
    combined["nightly_price_eur"],
    errors="coerce"
)

combined["total_price_eur"] = pd.to_numeric(
    combined["total_price_eur"],
    errors="coerce"
)

# Validate converted nightly prices

invalid_price_mask = (
    combined["nightly_price_eur"].isna()
    | (combined["nightly_price_eur"] <= 0)
    | (combined["nightly_price_eur"] == float("inf"))
)

if invalid_price_mask.any():
    raise ValueError(
        "Invalid converted nightly prices detected."
    )

print(
    f"Currency conversion time: "
    f"{time.perf_counter() - currency_start:.2f} seconds"
)


# 12. Display final preprocessing summary

print("\n--- PREPROCESSING SUMMARY ---")

print(f"Original records: {original_records:,}")
print(f"Remaining records: {len(combined):,}")
print(f"Total records removed: {original_records - len(combined):,}")

print(
    f"Percentage retained: "
    f"{len(combined) / original_records * 100:.2f}%"
)

print(f"Locations: {combined['city'].nunique()}")
print(f"Columns: {len(combined.columns)}")
print(f"Currencies converted: {combined['currency_resolved'].nunique()}")

print(
    f"Records with inferred currencies: "
    f"{combined['currency_imputed'].sum():,}"
)

print(
    f"Quotes at least 183 days in advance: "
    f"{(combined['booking_lead_days'] >= 183).sum():,}"
)

print("\nNightly price statistics (EUR):")

print(
    combined["nightly_price_eur"].describe(
        percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
    ).round(2)
)


# 13. Save preprocessed dataset

print("\n--- SAVING DATASET ---")

save_start = time.perf_counter()

combined.to_csv(output_file, index=False)

print(f"Dataset saved to: {output_file}")

print(
    f"Saving time: "
    f"{time.perf_counter() - save_start:.2f} seconds"
)

print(
    f"Total execution time: "
    f"{time.perf_counter() - start_time:.2f} seconds"
)

print("\nPreprocessing completed successfully.")