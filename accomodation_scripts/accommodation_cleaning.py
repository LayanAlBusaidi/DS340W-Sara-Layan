from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
import time

# Required packages:
# pip install pandas numpy

start_time = time.perf_counter()


# 1. Set file paths

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "accommodation" / "clean"

INPUT_FILE = DATA_DIR / "combined_accommodation.csv"
OUTPUT_FILE = DATA_DIR / "clean_accommodation.csv"


# 2. Load preprocessed accommodation dataset

print("\n--- LOADING ACCOMMODATION DATA ---")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Combined accommodation dataset not found."
    )

df = pd.read_csv(
    INPUT_FILE,
    dtype={"id": "string", "host_id": "string"},
    low_memory=False
)

original_records = len(df)
original_columns = len(df.columns)

print(f"Original records: {original_records:,}")
print(f"Original columns: {original_columns}")
print(f"Locations: {df['city'].nunique()}")

print(
    f"Loading time: "
    f"{time.perf_counter() - start_time:.2f} seconds"
)


# 3. Correct numerical and date types

numeric_columns = [
    "latitude",
    "longitude",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "reviews_per_month",
    "price_quote_total_price",
    "price_quote_price_per_night",
    "exchange_rate_to_eur",
    "nightly_price_eur",
    "total_price_eur",
    "booking_lead_days",
    "stay_duration",
    "checkin_month",
    "checkin_year"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df[column] = df[column].replace(
        [np.inf, -np.inf],
        np.nan
    )

date_columns = [
    "last_scraped",
    "price_quote_checkin_date",
    "price_quote_checkout_date"
]

for column in date_columns:

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )


# 4. Validate temporal features

print("\n--- DATA VALIDATION ---")

# Recalculate temporal features from original dates

df["booking_lead_days"] = (
    df["price_quote_checkin_date"]
    - df["last_scraped"]
).dt.days

df["stay_duration"] = (
    df["price_quote_checkout_date"]
    - df["price_quote_checkin_date"]
).dt.days

df["checkin_month"] = (
    df["price_quote_checkin_date"].dt.month
)

df["checkin_year"] = (
    df["price_quote_checkin_date"].dt.year
)


# Identify invalid records

invalid_price = (
    df["nightly_price_eur"].isna()
    | (df["nightly_price_eur"] <= 0)
)

invalid_lead_time = (
    df["booking_lead_days"].isna()
    | (df["booking_lead_days"] < 0)
)

invalid_stay = (
    df["stay_duration"].isna()
    | (df["stay_duration"] <= 0)
)

invalid_capacity = (
    df["accommodates"].isna()
    | (df["accommodates"] <= 0)
)

invalid_id = (
    df["id"].isna()
    | df["id"].str.strip().eq("").fillna(False)
)

print(f"Invalid nightly prices: {invalid_price.sum():,}")
print(f"Invalid booking lead times: {invalid_lead_time.sum():,}")
print(f"Invalid stay durations: {invalid_stay.sum():,}")
print(f"Invalid guest capacities: {invalid_capacity.sum():,}")
print(f"Missing listing IDs: {invalid_id.sum():,}")

# Remove invalid records

invalid_records = (
    invalid_price
    | invalid_lead_time
    | invalid_stay
    | invalid_capacity
    | invalid_id
)

print(
    f"Total invalid records removed: "
    f"{invalid_records.sum():,}"
)

df = df.loc[~invalid_records].copy()


# 5. Standardize boolean features

def normalize_boolean(series):

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({
            "true": 1,
            "false": 0,
            "t": 1,
            "f": 0,
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "1.0": 1,
            "0.0": 0
        })
        .astype("Int8")
    )


boolean_columns = [
    "host_is_superhost",
    "quote_available",
    "is_monthly_priced"
]

for column in boolean_columns:

    df[column] = normalize_boolean(df[column])


# 6. Validate quote availability

print("\n--- QUOTE AVAILABILITY ---")

# Remove explicitly unavailable quotes
# Preserve quotes with unknown availability

unavailable_mask = (
    df["quote_available"].eq(0).fillna(False)
)

print(
    f"Unavailable quotes removed: "
    f"{unavailable_mask.sum():,}"
)

df = df.loc[~unavailable_mask].copy()

df["quote_availability_unknown"] = (
    df["quote_available"].isna().astype("int8")
)

print(
    f"Quotes with unknown availability: "
    f"{df['quote_availability_unknown'].sum():,}"
)


# 7. Validate property features

print("\n--- PROPERTY VALIDATION ---")

# Negative property values are invalid.
# Zero bedrooms may represent a studio or other room type.

nonnegative_columns = [
    "bedrooms",
    "beds",
    "bathrooms",
    "number_of_reviews",
    "reviews_per_month"
]

for column in nonnegative_columns:

    invalid = df[column] < 0

    if invalid.any():

        print(
            f"Invalid {column} values: "
            f"{invalid.sum():,}"
        )

        df.loc[invalid, column] = np.nan


# Validate minimum and maximum stay restrictions

for column in ["minimum_nights", "maximum_nights"]:

    invalid = df[column] <= 0

    if invalid.any():

        print(
            f"Invalid {column} values: "
            f"{invalid.sum():,}"
        )

        df.loc[invalid, column] = np.nan


# Check inconsistent minimum and maximum stay restrictions

invalid_restrictions = (
    df["minimum_nights"].notna()
    & df["maximum_nights"].notna()
    & (df["maximum_nights"] < df["minimum_nights"])
)

print(
    f"Inconsistent stay restrictions: "
    f"{invalid_restrictions.sum():,}"
)

# Preserve records but mark inconsistent restrictions as missing

df.loc[
    invalid_restrictions,
    ["minimum_nights", "maximum_nights"]
] = np.nan


# Identify quotes that violate listing stay restrictions
# Retain these observations because listing restrictions
# may differ from date-specific quote restrictions

df["below_minimum_stay"] = (
    df["stay_duration"] < df["minimum_nights"]
).where(
    df["minimum_nights"].notna()
).astype("Int8")

df["above_maximum_stay"] = (
    df["stay_duration"] > df["maximum_nights"]
).where(
    df["maximum_nights"].notna()
).astype("Int8")

print(
    f"Quotes below minimum stay: "
    f"{df['below_minimum_stay'].sum():,}"
)

print(
    f"Quotes above maximum stay: "
    f"{df['above_maximum_stay'].sum():,}"
)


# Validate availability counts

availability_columns = {
    "availability_30": 30,
    "availability_60": 60,
    "availability_90": 90,
    "availability_365": 365
}

for column, maximum in availability_columns.items():

    invalid = (
        (df[column] < 0)
        | (df[column] > maximum)
    )

    if invalid.any():

        print(
            f"Invalid {column} values: "
            f"{invalid.sum():,}"
        )

        df.loc[invalid, column] = np.nan


# Validate geographical coordinates

df.loc[
    ~df["latitude"].between(-90, 90),
    "latitude"
] = np.nan

df.loc[
    ~df["longitude"].between(-180, 180),
    "longitude"
] = np.nan


# 8. Identify and remove duplicate observations

print("\n--- DUPLICATE ANALYSIS ---")

duplicate_keys = [
    "id",
    "last_scraped",
    "price_quote_checkin_date",
    "price_quote_checkout_date"
]

duplicate_mask = df.duplicated(
    subset=duplicate_keys,
    keep=False
)

print(
    f"Records sharing listing and quote keys: "
    f"{duplicate_mask.sum():,}"
)

# Identify repeated quotes with conflicting prices

duplicate_prices = (
    df.loc[duplicate_mask]
    .groupby(duplicate_keys)["nightly_price_eur"]
    .nunique()
)

conflicting_keys = duplicate_prices[
    duplicate_prices > 1
]

print(
    f"Duplicate keys with conflicting prices: "
    f"{len(conflicting_keys):,}"
)

# Remove identical-price duplicate quotes
# Preserve conflicting-price observations for investigation

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=duplicate_keys + ["nightly_price_eur"],
    keep="first"
).copy()

print(
    f"Duplicate observations removed: "
    f"{before_duplicates - len(df):,}"
)


# 9. Apply domain-based price filtering

print("\n--- DOMAIN-BASED PRICE FILTERING ---")

# Define acceptable nightly accommodation prices in EUR

MIN_NIGHTLY_PRICE = 10
MAX_NIGHTLY_PRICE = 5000

before_filtering = len(df)

outlier_mask = ~df["nightly_price_eur"].between(
    MIN_NIGHTLY_PRICE,
    MAX_NIGHTLY_PRICE
)

below_minimum = (
    df["nightly_price_eur"] < MIN_NIGHTLY_PRICE
).sum()

above_maximum = (
    df["nightly_price_eur"] > MAX_NIGHTLY_PRICE
).sum()

extended_removed = (
    outlier_mask
    & (df["booking_lead_days"] >= 183)
).sum()

print(
    f"Accepted nightly price range: "
    f"EUR {MIN_NIGHTLY_PRICE:,} - {MAX_NIGHTLY_PRICE:,}"
)

print(f"Prices below minimum: {below_minimum:,}")
print(f"Prices above maximum: {above_maximum:,}")
print(f"Total price outliers removed: {outlier_mask.sum():,}")
print(f"Extended-horizon quotes removed: {extended_removed:,}")


# Analyze filtering impact across locations and room types

for column in ["city", "room_type"]:

    impact = (
        outlier_mask.groupby(df[column], dropna=False)
        .agg(["sum", "size"])
    )

    impact.columns = [
        "removed",
        "original"
    ]

    impact["removed_pct"] = (
        impact["removed"]
        / impact["original"]
        * 100
    ).round(2)

    impact = (
        impact.loc[impact["removed"] > 0]
        .sort_values("removed_pct", ascending=False)
    )

    print(f"\nHighest removal rates by {column}:")

    print(
        impact.head(5).to_string()
    )


# Apply filtering

before_locations = df["city"].nunique()

df = df.loc[~outlier_mask].copy()

print(
    f"\nLocations lost: "
    f"{before_locations - df['city'].nunique()}"
)

print(f"Remaining records: {len(df):,}")

print(
    f"Percentage retained after price filtering: "
    f"{len(df) / before_filtering * 100:.2f}%"
)


# 10. Handle missing categorical values

print("\n--- MISSING VALUE HANDLING ---")

categorical_columns = [
    "city",
    "neighbourhood_cleansed",
    "property_type",
    "room_type"
]

for column in categorical_columns:

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("Unknown")
    )


# 11. Handle review information

# Keep listings without reviews

df["has_reviews"] = (
    df["number_of_reviews"] > 0
).where(
    df["number_of_reviews"].notna()
).astype("Int8")

# Preserve missing rating information

df["review_rating_missing"] = (
    df["review_scores_rating"].isna()
    .astype("int8")
)

# Missing monthly review counts can be set to zero
# when the listing has no reviews

no_reviews_mask = (
    (df["number_of_reviews"] == 0)
    & df["reviews_per_month"].isna()
)

df.loc[
    no_reviews_mask,
    "reviews_per_month"
] = 0

print(
    f"Missing monthly reviews replaced with zero: "
    f"{no_reviews_mask.sum():,}"
)

print(
    f"Listings without review ratings: "
    f"{df['review_rating_missing'].sum():,}"
)

# Remaining missing ratings will be handled
# during model development


# 12. Extract bathroom features

print("\n--- BATHROOM FEATURE ENGINEERING ---")

bathroom_text = (
    df["bathrooms_text"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Extract numerical bathroom counts

bathroom_numbers = pd.to_numeric(
    bathroom_text.str.extract(
        r"(\d+(?:\.\d+)?)",
        expand=False
    ),
    errors="coerce"
)

# Handle half-bath descriptions

half_bath_mask = bathroom_text.str.contains(
    r"half[\s-]*bath",
    regex=True,
    na=False
)

bathroom_numbers.loc[
    half_bath_mask & bathroom_numbers.isna()
] = 0.5

# Preserve existing numerical values and recover missing ones

df["bathroom_count"] = (
    df["bathrooms"].fillna(bathroom_numbers)
)

# Identify shared bathrooms

df["shared_bathroom"] = (
    bathroom_text.str.contains(
        "shared",
        regex=False
    )
    .where(bathroom_text.notna())
    .astype("Int8")
)

# Preserve missing-value information

df["bathroom_count_missing"] = (
    df["bathroom_count"].isna()
    .astype("int8")
)

recovered_bathrooms = (
    df["bathrooms"].isna()
    & df["bathroom_count"].notna()
)

print(
    f"Bathroom counts recovered: "
    f"{recovered_bathrooms.sum():,}"
)

print(
    f"Remaining missing bathroom counts: "
    f"{df['bathroom_count'].isna().sum():,}"
)


# 13. Extract amenity features

print("\n--- AMENITY FEATURE ENGINEERING ---")

amenity_start = time.perf_counter()

# Regular expressions and word boundaries reduce
# incorrect matches between similar amenity descriptions

AMENITY_PATTERNS = {
    "has_wifi": r"\b(?:wi[ -]?fi|wireless internet)\b",
    "has_kitchen": r"\b(?:kitchen|kitchenette)\b",
    "has_air_conditioning": r"\bair conditioning\b",
    "has_heating": r"\bheating\b",
    "has_pool": r"\bpool\b",
    "has_parking": r"\bparking\b",
    "has_washer": r"\b(?:washer|washing machine)\b",
    "has_dryer": r"\b(?:dryer|tumble dryer)\b",
    "has_tv": r"\b(?:tv|television|hdtv)\b",
    "has_workspace": r"\bdedicated workspace\b",
    "has_gym": r"\b(?:gym|fitness cent(?:er|re))\b",
    "has_hot_tub": r"\b(?:hot tub|jacuzzi)\b",
    "has_breakfast": r"\bbreakfast\b",
    "has_elevator": r"\b(?:elevator|lift)\b"
}


# Exclude descriptions that could cause false matches

AMENITY_EXCLUSIONS = {
    "has_pool": (
        r"\b(?:pool table|pool toys|pool cover|"
        r"pool towels|pool fence|pool access fee)\b"
    ),
    "has_dryer": r"\bhair[ -]?dryer\b",
    "has_heating": r"\b(?:heating pad|heating blanket)\b",
    "has_breakfast": (
        r"\b(?:breakfast table|breakfast bar|breakfast nook)\b"
    ),
    "has_elevator": r"\b(?:ski lift|chair lift)\b"
}


# Compile patterns once for repeated use

compiled_patterns = {
    name: re.compile(pattern)
    for name, pattern in AMENITY_PATTERNS.items()
}

compiled_exclusions = {
    name: re.compile(pattern)
    for name, pattern in AMENITY_EXCLUSIONS.items()
}

# Exclude explicitly unavailable amenities

negative_pattern = re.compile(
    r"^(?:no|not available|without)\b"
)


amenity_columns = [
    "amenity_count",
    "amenities_missing"
] + list(AMENITY_PATTERNS.keys())


def extract_amenities(value):

    # Return missing values when amenities cannot be read

    missing_result = (
        -1,
        1,
        *([-1] * len(AMENITY_PATTERNS))
    )

    if not isinstance(value, str):
        return missing_result

    try:

        amenities = json.loads(value)

        if not isinstance(amenities, list):
            return missing_result

        # Normalize and remove duplicate amenities

        amenities = {
            item.strip().lower()
            for item in amenities
            if isinstance(item, str) and item.strip()
        }

        features = [
            len(amenities),
            0
        ]

        # Identify individual amenities

        for name, pattern in compiled_patterns.items():

            exclusion = compiled_exclusions.get(name)

            available = any(
                pattern.search(amenity)
                and not negative_pattern.search(amenity)
                and not (
                    exclusion
                    and exclusion.search(amenity)
                )
                for amenity in amenities
            )

            features.append(int(available))

        return tuple(features)

    except (TypeError, ValueError):

        return missing_result


# Extract all amenity features in one pass

amenity_array = np.empty(
    (len(df), len(amenity_columns)),
    dtype=np.int32
)

for i, value in enumerate(df["amenities"]):

    amenity_array[i] = extract_amenities(value)

    if (i + 1) % 250000 == 0:

        print(
            f"Listings processed: "
            f"{i + 1:,}/{len(df):,}"
        )


# Add extracted features to the dataset

for i, column in enumerate(amenity_columns):

    df[column] = amenity_array[:, i]

    # Replace temporary missing-value marker

    df[column] = df[column].replace(
        -1,
        np.nan
    )

del amenity_array

print(f"Amenity features created: {len(amenity_columns)}")

print(
    f"Missing or unreadable amenity records: "
    f"{df['amenities_missing'].sum():,}"
)


# Verify the distribution of extracted amenity features

print("\nAmenity prevalence (% of readable records):")

amenity_summary = (
    df[list(AMENITY_PATTERNS.keys())].mean() * 100
).round(2)

print(amenity_summary.to_string())

print(
    f"\nAmenity processing time: "
    f"{time.perf_counter() - amenity_start:.2f} seconds"
)


# 14. Create additional temporal features

print("\n--- TEMPORAL FEATURE ENGINEERING ---")

df["checkin_day_of_week"] = (
    df["price_quote_checkin_date"].dt.dayofweek
)

# Friday and Saturday are treated as weekend arrivals
# Monday = 0, Friday = 4, Saturday = 5

df["is_weekend"] = (
    df["checkin_day_of_week"].isin([4, 5])
    .astype("int8")
)

df["checkin_quarter"] = (
    df["price_quote_checkin_date"].dt.quarter
)


# Cyclical encoding of month

df["month_sin"] = np.sin(
    2 * np.pi * (df["checkin_month"] - 1) / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * (df["checkin_month"] - 1) / 12
)


# Cyclical encoding of weekday

df["weekday_sin"] = np.sin(
    2 * np.pi * df["checkin_day_of_week"] / 7
)

df["weekday_cos"] = np.cos(
    2 * np.pi * df["checkin_day_of_week"] / 7
)


# Categorize booking horizons

df["booking_horizon"] = pd.cut(
    df["booking_lead_days"],
    bins=[-1, 7, 30, 90, 182, np.inf],
    labels=[
        "immediate",
        "short_term",
        "medium_term",
        "long_term",
        "extended"
    ]
)

print("Booking horizon distribution:")

print(
    df["booking_horizon"]
    .value_counts(sort=False)
    .to_string()
)


# 15. Create additional property features

# Identify listings with zero bedrooms

df["has_zero_bedrooms"] = (
    df["bedrooms"] == 0
).where(
    df["bedrooms"].notna()
).astype("Int8")


# Calculate guest capacity relative to bedrooms

df["guests_per_bedroom"] = (
    df["accommodates"]
    / df["bedrooms"].replace(0, np.nan)
)


# Calculate guest capacity relative to beds

df["guests_per_bed"] = (
    df["accommodates"]
    / df["beds"].replace(0, np.nan)
)


# Preserve missing-value indicators

df["bedrooms_missing"] = (
    df["bedrooms"].isna().astype("int8")
)

df["beds_missing"] = (
    df["beds"].isna().astype("int8")
)


# 16. Remove unnecessary columns

print("\n--- COLUMN CLEANING ---")

columns_to_remove = [
    "instant_bookable",
    "bathrooms",
    "bathrooms_text",
    "amenities"
]

before_column_removal = len(df.columns)

df = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)

print(
    f"Columns removed: "
    f"{before_column_removal - len(df.columns)}"
)


# 17. Identify columns excluded from model inputs

# Retain these columns for documentation, validation,
# data splitting, and reproducibility.
# They should not be passed directly into the models.

EXCLUDED_MODEL_COLUMNS = [
    "id",
    "host_id",
    "source_file",
    "last_scraped",
    "scrape_date",
    "price_quote_checkin_date",
    "price_quote_checkout_date",
    "currency",
    "currency_resolved",
    "currency_imputed",
    "exchange_rate_to_eur",
    "exchange_rate_date",
    "price",
    "price_quote_price_per_night",
    "price_quote_total_price",
    "total_price_eur",
    "nightly_price_eur",
    "quote_available",
    "quote_availability_unknown",
    "is_monthly_priced",
    "below_minimum_stay",
    "above_maximum_stay"
]

candidate_features = [
    column for column in df.columns
    if column not in EXCLUDED_MODEL_COLUMNS
]


# Identify constant candidate features

constant_features = [
    column for column in candidate_features
    if df[column].nunique(dropna=False) <= 1
]

print(f"Constant features identified: {len(constant_features)}")

if constant_features:

    print(
        ", ".join(constant_features)
    )


# Exclude constant features from preliminary model candidates
# Retain them in the dataset for documentation

candidate_features = [
    column for column in candidate_features
    if column not in constant_features
]

print(
    f"Candidate model features: "
    f"{len(candidate_features)}"
)

# Constant-feature detection will be repeated using
# training data during model development.

# Final feature selection will also consider which
# features are available when making future predictions.


# 18. Final dataset validation

print("\n--- FINAL DATASET VALIDATION ---")

# Ensure the dataset is not empty

assert not df.empty

# Validate target values

assert df["nightly_price_eur"].notna().all()

assert np.isfinite(
    df["nightly_price_eur"]
).all()

assert df["nightly_price_eur"].between(
    MIN_NIGHTLY_PRICE,
    MAX_NIGHTLY_PRICE
).all()

# Validate temporal features

assert (df["booking_lead_days"] >= 0).all()

assert (df["stay_duration"] > 0).all()

# Validate guest capacity

assert (df["accommodates"] > 0).all()

print("Target and temporal validation passed.")


# Summarize remaining missing candidate features

missing_summary = (
    df[candidate_features].isna().mean() * 100
).sort_values(ascending=False)

missing_summary = missing_summary[
    missing_summary > 0
]

print("\nRemaining missing candidate features (%):")

print(
    missing_summary.round(2).to_string()
)

# Statistical imputation will be performed after
# splitting the data to prevent data leakage.


# 19. Display final cleaning summary

print("\n--- CLEANING SUMMARY ---")

print(f"Original records: {original_records:,}")
print(f"Remaining records: {len(df):,}")

print(
    f"Total records removed: "
    f"{original_records - len(df):,}"
)

print(
    f"Percentage retained: "
    f"{len(df) / original_records * 100:.2f}%"
)

print(f"Locations: {df['city'].nunique()}")
print(f"Original columns: {original_columns}")
print(f"Final columns: {len(df.columns)}")

print(
    f"Candidate model features: "
    f"{len(candidate_features)}"
)

print(
    f"Quotes at least 183 days in advance: "
    f"{(df['booking_lead_days'] >= 183).sum():,}"
)


# Display final nightly price distribution

print("\nFinal nightly price statistics (EUR):")

print(
    df["nightly_price_eur"].describe(
        percentiles=[
            0.01,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ]
    ).round(2)
)


# 20. Save cleaned accommodation dataset

print("\n--- SAVING DATASET ---")

save_start = time.perf_counter()

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Dataset saved to: {OUTPUT_FILE}")

print(
    f"Saving time: "
    f"{time.perf_counter() - save_start:.2f} seconds"
)

print(
    f"Total execution time: "
    f"{time.perf_counter() - start_time:.2f} seconds"
)

print("\nCleaning and feature engineering completed successfully.")