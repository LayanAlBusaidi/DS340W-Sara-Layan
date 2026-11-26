# Required packages:
# python -m pip install pandas numpy scikit-learn joblib

from pathlib import Path
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# 1. Set paths and configuration

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "accommodation" / "clean" / "clean_accommodation.csv"

RESULTS_DIR = ROOT / "accomodation_scripts" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "nightly_price_eur"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# 2. Define feature groups

# Basic trip and accommodation features
BASIC_FEATURES = [
    "city",
    "room_type",
    "accommodates",
    "stay_duration"
]

# Temporal and seasonal features
TEMPORAL_FEATURES = [
    "booking_lead_days",
    "checkin_month",
    "checkin_year",
    "checkin_quarter",
    "checkin_day_of_week",
    "is_weekend",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "booking_horizon"
]

# Location features
LOCATION_FEATURES = [
    "neighbourhood_cleansed",
    "latitude",
    "longitude"
]

# Property features
PROPERTY_FEATURES = [
    "property_type",
    "bedrooms",
    "beds",
    "bathroom_count",
    "shared_bathroom",
    "has_zero_bedrooms",
    "guests_per_bedroom",
    "guests_per_bed"
]

# Accommodation amenities
AMENITY_FEATURES = [
    "amenity_count",
    "has_wifi",
    "has_kitchen",
    "has_air_conditioning",
    "has_heating",
    "has_pool",
    "has_parking",
    "has_washer",
    "has_dryer",
    "has_tv",
    "has_workspace",
    "has_gym",
    "has_hot_tub",
    "has_breakfast",
    "has_elevator"
]

# Review and host features
REVIEW_FEATURES = [
    "number_of_reviews",
    "review_scores_rating",
    "reviews_per_month",
    "host_is_superhost",
    "has_reviews"
]

# Availability and stay restrictions
AVAILABILITY_FEATURES = [
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "minimum_nights",
    "maximum_nights"
]

# Missing value indicators
MISSING_VALUE_FEATURES = [
    "bedrooms_missing",
    "beds_missing",
    "bathroom_count_missing",
    "review_rating_missing"
]


# 3. Define experimental feature sets

FEATURE_SETS = {
    "basic": BASIC_FEATURES,

    "temporal": (
        BASIC_FEATURES
        + TEMPORAL_FEATURES
    ),

    "comprehensive": (
        BASIC_FEATURES
        + TEMPORAL_FEATURES
        + LOCATION_FEATURES
        + PROPERTY_FEATURES
        + AMENITY_FEATURES
        + REVIEW_FEATURES
        + AVAILABILITY_FEATURES
        + MISSING_VALUE_FEATURES
    )
}

# Features that require categorical encoding
CATEGORICAL_FEATURES = [
    "city",
    "room_type",
    "neighbourhood_cleansed",
    "property_type",
    "booking_horizon"
]


# 4. Define excluded features

# Exclude identifiers, redundant columns and potential leakage
EXCLUDED_FEATURES = [
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
    "above_maximum_stay",
    "amenities_missing"
]


# 5. Load cleaned data

def load_data():

    start_time = time.time()

    print("\n--- Loading cleaned accommodation data ---")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    # Load required features, target and listing information
    required_columns = list(dict.fromkeys(
        FEATURE_SETS["comprehensive"]
        + [TARGET, "id", "last_scraped"]
    ))

    df = pd.read_csv(
        DATA_FILE,
        usecols=required_columns,
        dtype={"id": "string"},
        low_memory=False
    )

    # Correct date types
    df["last_scraped"] = pd.to_datetime(
        df["last_scraped"],
        errors="coerce"
    )

    # Correct categorical types and fill missing categories
    for column in CATEGORICAL_FEATURES:

        df[column] = (
            df[column]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
        )

    # Identify numerical features
    numerical_features = [
        column
        for column in FEATURE_SETS["comprehensive"]
        if column not in CATEGORICAL_FEATURES
    ]

    # Correct numerical types and replace infinite values
    for column in numerical_features:

        df[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    print(f"Total observations: {len(df):,}")
    print(f"Loaded columns: {len(df.columns)}")
    print(f"Locations: {df['city'].nunique():,}")
    print(f"Unique listing IDs: {df['id'].nunique():,}")
    print(f"Loading time: {time.time() - start_time:.2f} seconds")

    return df


# 6. Validate data

def validate_data(df):

    print("\n--- Data validation ---")

    if df.empty:
        raise ValueError("The dataset is empty.")

    # Check target values
    if df[TARGET].isna().any():
        raise ValueError("Missing target values detected.")

    if not np.isfinite(df[TARGET]).all():
        raise ValueError("Invalid target values detected.")

    if not df[TARGET].between(10, 5000).all():
        raise ValueError("Target prices outside accepted range.")

    # Check listing IDs and dates
    if df["id"].isna().any():
        raise ValueError("Missing listing IDs detected.")

    if df["last_scraped"].isna().any():
        raise ValueError("Missing collection dates detected.")

    # Check booking lead times
    if df["booking_lead_days"].isna().any():
        raise ValueError("Missing booking lead times detected.")

    if (df["booking_lead_days"] < 0).any():
        raise ValueError("Negative booking lead times detected.")

    # Check stay durations
    if (
        df["stay_duration"].isna().any()
        or (df["stay_duration"] <= 0).any()
    ):
        raise ValueError("Invalid stay durations detected.")

    # Check feature availability and potential leakage
    for name, features in FEATURE_SETS.items():

        missing_features = [
            feature
            for feature in features
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing features in {name}: {missing_features}"
            )

        leakage_features = (
            set(features) & set(EXCLUDED_FEATURES)
        )

        if leakage_features:
            raise ValueError(
                f"Excluded features in {name}: {leakage_features}"
            )

    print("Target validation passed.")
    print("Feature validation passed.")
    print("Leakage exclusion checks passed.")


# 7. Inspect dataset structure

def inspect_dataset(df):

    print("\n--- Dataset structure ---")

    print(
        "Earliest collection date:",
        df["last_scraped"].min()
    )

    print(
        "Latest collection date:",
        df["last_scraped"].max()
    )

    print(
        "Unique collection dates:",
        df["last_scraped"].dt.normalize().nunique()
    )

    # Count listings with multiple observations
    repeated_listings = (
        df["id"]
        .value_counts()
        .gt(1)
        .sum()
    )

    print(
        f"Listing IDs with multiple observations: "
        f"{repeated_listings:,}"
    )

    print(
        f"Maximum booking lead time: "
        f"{df['booking_lead_days'].max():.0f} days"
    )

    # Display feature set sizes
    print("\nFeature set sizes:")

    for name, features in FEATURE_SETS.items():
        print(f"{name}: {len(features)} features")


# 8. Split data into training, validation and testing sets

def split_data(df):

    print("\n--- Data splitting ---")

    # Group observations by listing ID
    groups = df["id"]

    # Separate 15% for testing
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_val_indices, test_indices = next(
        first_split.split(df, groups=groups)
    )

    train_val_df = df.iloc[train_val_indices]
    test_df = df.iloc[test_indices]

    # Split remaining data into training and validation
    relative_validation_size = (
        VALIDATION_SIZE / (TRAIN_SIZE + VALIDATION_SIZE)
    )

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_validation_size,
        random_state=RANDOM_STATE
    )

    train_indices, validation_indices = next(
        second_split.split(
            train_val_df,
            groups=train_val_df["id"]
        )
    )

    train_df = train_val_df.iloc[train_indices]
    validation_df = train_val_df.iloc[validation_indices]

    # Check for overlapping listing IDs
    train_ids = set(train_df["id"])
    validation_ids = set(validation_df["id"])
    test_ids = set(test_df["id"])

    assert not train_ids.intersection(validation_ids)
    assert not train_ids.intersection(test_ids)
    assert not validation_ids.intersection(test_ids)

    print("Listing overlap validation passed.")

    # Check that all observations are included
    assert (
        len(train_df)
        + len(validation_df)
        + len(test_df)
        == len(df)
    )

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df
    }

    # Display split sizes
    for name, split in splits.items():

        percentage = len(split) / len(df) * 100

        print(
            f"{name.capitalize()}: "
            f"{len(split):,} observations "
            f"({percentage:.2f}%)"
        )

    return splits


# 9. Check split distributions

def inspect_splits(splits):

    print("\n--- Split distribution analysis ---")

    summary = []

    # Summarize each split
    for name, df in splits.items():

        extended_count = (
            df["booking_lead_days"] >= 183
        ).sum()

        summary.append({
            "split": name,
            "observations": len(df),
            "unique_listings": df["id"].nunique(),
            "locations": df["city"].nunique(),
            "extended_horizon": extended_count,
            "extended_horizon_pct": (
                extended_count / len(df) * 100
            )
        })

    summary_df = pd.DataFrame(summary)

    print(summary_df.round(2).to_string(index=False))

    # Compare booking horizon distributions
    print("\nBooking horizon distribution:")

    horizon_distribution = pd.DataFrame({
        name: (
            df["booking_horizon"]
            .value_counts(normalize=True) * 100
        )
        for name, df in splits.items()
    })

    print(horizon_distribution.round(2))

    # Check for cities not represented in training
    training_cities = set(splits["train"]["city"])

    for name in ["validation", "test"]:

        unseen_cities = (
            set(splits[name]["city"]) - training_cities
        )

        print(
            f"\nUnseen cities in {name}: "
            f"{len(unseen_cities)}"
        )

        if unseen_cities:
            print(sorted(unseen_cities))


# 10. Inspect training features

def inspect_training_features(train_df):

    print("\n--- Training feature analysis ---")

    features = FEATURE_SETS["comprehensive"]

    # Calculate missing value percentages
    missing_values = (
        train_df[features]
        .isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    missing_values = missing_values[missing_values > 0]

    print("\nMissing feature values (%):")
    print(missing_values.round(2))

    # Identify constant features
    constant_features = [
        column
        for column in features
        if train_df[column].nunique(dropna=False) <= 1
    ]

    print(
        f"\nConstant training features: "
        f"{len(constant_features)}"
    )

    if constant_features:
        print(constant_features)

    # Display target statistics
    print("\nTraining target statistics:")
    print(train_df[TARGET].describe().round(2))


# 11. Prepare model inputs

def prepare_model_data(splits, feature_set="comprehensive"):

    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )

    selected_features = FEATURE_SETS[feature_set].copy()

    # Remove constant features using training data
    selected_features = [
        column
        for column in selected_features
        if splits["train"][column].nunique(dropna=False) > 1
    ]

    # Separate categorical and numerical features
    categorical_features = [
        column
        for column in selected_features
        if column in CATEGORICAL_FEATURES
    ]

    numerical_features = [
        column
        for column in selected_features
        if column not in CATEGORICAL_FEATURES
    ]

    prepared_data = {}

    # Separate independent and dependent variables
    for name, df in splits.items():

        prepared_data[f"X_{name}"] = df[selected_features]
        prepared_data[f"y_{name}"] = df[TARGET]

    # Store selected feature information
    prepared_data["features"] = selected_features
    prepared_data["numerical_features"] = numerical_features
    prepared_data["categorical_features"] = categorical_features

    print(f"\nSelected feature set: {feature_set}")
    print(f"Total features: {len(selected_features)}")
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    return prepared_data


# 12. Create preprocessing pipeline

def create_preprocessor(
    numerical_features,
    categorical_features,
    scale_numeric=True
):

    # Fill missing numerical values using the median
    numerical_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    # Apply scaling when required by the model
    if scale_numeric:
        numerical_steps.append(
            ("scaler", StandardScaler())
        )

    numerical_pipeline = Pipeline(
        steps=numerical_steps
    )

    # Fill missing categories and apply One-Hot Encoding
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=True
                )
            )
        ]
    )

    # Combine numerical and categorical preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        sparse_threshold=1.0
    )

    return preprocessor