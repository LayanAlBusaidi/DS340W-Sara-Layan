
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import streamlit as st

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT / "results" / "recurrent_flight_models"
    / "rnn_model.pt"
)

DATA_PATH = (
    ROOT / "data" / "fares_by_month_cleaned.csv"
)

# 1. Reconstruct the original RNN architecture
class RecurrentFareModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.recurrent = nn.RNN(
            input_size=1,
            hidden_size=32,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, _ = self.recurrent(x)

        last_step = output[:, -1, :]

        return self.head(last_step).squeeze(1)


# 2. Load the trained model
@st.cache_resource
def load_airfare_model():

    model = RecurrentFareModel()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    return model


# 3. Recover the original normalization values
@st.cache_data
def load_normalization():

    train = pd.read_csv(
        ROOT / "data" / "processed" / "train.csv"
    )

    val = pd.read_csv(
        ROOT / "data" / "processed" / "validation.csv"
    )

    df = pd.concat([train, val], ignore_index=True)

    df["departure_month"] = pd.to_datetime(
        df["departure_month"]
    ).dt.to_period("M")

    df["route"] = (
        df["origin_city"].astype(str)
        + " -> "
        + df["destination_city"].astype(str)
    )

    lookup = {
        (row.route, row.departure_month): row.median_eur
        for row in df.itertuples(index=False)
    }

    target_month = pd.Period("2026-11", freq="M")

    targets = []

    for row in df[
        df["departure_month"] == target_month
    ].itertuples(index=False):

        if (
            (row.route, target_month - 2) in lookup
            and
            (row.route, target_month - 1) in lookup
        ):
            targets.append(row.median_eur)

    targets = np.asarray(
        targets,
        dtype=np.float32
    )

    if len(targets) == 0:
        raise ValueError("Training targets not found.")

    scale_mean = float(targets.mean())
    scale_std = float(targets.std())

    if not np.isfinite(scale_std) or scale_std <= 0:
        raise ValueError("Invalid normalization scale.")

    return scale_mean, scale_std


# 4. Load available airfare records
@st.cache_data
def load_airfare_data():

    df = pd.read_csv(DATA_PATH)

    df["departure_month"] = pd.to_datetime(
        df["departure_month"]
    ).dt.to_period("M")

    df = df.dropna(
        subset=[
            "origin_city",
            "destination_city",
            "departure_month",
            "median_eur"
        ]
    )

    df["origin_city"] = df["origin_city"].str.strip()
    df["destination_city"] = (
        df["destination_city"].str.strip()
    )

    return df


# 5. Predict the next month's airfare
def predict_next_month(previous_2, previous_1):

    model = load_airfare_model()

    scale_mean, scale_std = load_normalization()

    sequence = np.array(
        [[[previous_2], [previous_1]]],
        dtype=np.float32
    )

    sequence = (
        sequence - scale_mean
    ) / scale_std

    inputs = torch.tensor(
        sequence,
        dtype=torch.float32
    )

    with torch.no_grad():
        prediction = model(inputs).item()

    prediction = (
        prediction * scale_std + scale_mean
    )

    return max(0.0, prediction)


# 6. Predict airfare for a specific route and month
def predict_one_way(origin, destination, travel_date):

    df = load_airfare_data()

    target_month = pd.Period(
        travel_date,
        freq="M"
    )

    route_data = df[
        (df["origin_city"] == origin)
        &
        (df["destination_city"] == destination)
        &
        (df["departure_month"] < target_month)
    ].sort_values("departure_month")

    if route_data.empty:
        raise ValueError(
            f"No historical data for "
            f"{origin} to {destination}."
        )

    if route_data["departure_month"].duplicated().any():
        raise ValueError("Duplicate route-month records.")

    fare_history = dict(zip(
        route_data["departure_month"],
        route_data["median_eur"]
    ))

    # Find the latest two consecutive available months
    valid_months = [
        month for month in fare_history
        if month - 1 in fare_history
    ]

    if not valid_months:
        raise ValueError(
            f"Insufficient consecutive monthly data "
            f"for {origin} to {destination}."
        )

    latest_month = max(valid_months)

    previous_2 = fare_history[latest_month - 1]
    previous_1 = fare_history[latest_month]

    current_month = latest_month + 1

    # Recursive forecasting
    while current_month <= target_month:

        if current_month in fare_history:
            prediction = fare_history[current_month]
        else:
            prediction = predict_next_month(
                previous_2,
                previous_1
            )

        previous_2 = previous_1
        previous_1 = prediction

        current_month += 1

    return float(previous_1)


# 7. Calculate total airfare
def predict_airfare(user_input):

    origin = user_input["origin"]
    destination = user_input["destination"]

    travelers = user_input["travelers"]

    outbound = predict_one_way(
        origin,
        destination,
        user_input["departure_date"]
    )

    inbound = predict_one_way(
        destination,
        origin,
        user_input["return_date"]
    )

    total = (outbound + inbound) * travelers

    return {
        "outbound": round(outbound, 2),
        "return": round(inbound, 2),
        "per_person": round(outbound + inbound, 2),
        "total": round(total, 2)
    }

@st.cache_data
def get_eligible_routes():

    df = load_airfare_data()

    eligible = []

    for (origin, destination), group in df.groupby(
        ["origin_city", "destination_city"]
    ):

        if origin == destination:
            continue

        months = set(group["departure_month"])

        if any(month - 1 in months for month in months):
            eligible.append((origin, destination))

    eligible_set = set(eligible)

    round_trip_routes = [
        (origin, destination)
        for origin, destination in eligible
        if (destination, origin) in eligible_set
    ]

    return pd.DataFrame(
        round_trip_routes,
        columns=["origin_city", "destination_city"]
    )


# ==========================================
# DAILY EXPENSE ESTIMATION
# ==========================================

import math
import unicodedata

DAILY_COST_PATH = (
    ROOT / "data" / "daily_cost" / "clean"
    / "daily_cost_estimated.csv"
)

# Temporary fixed currency conversion rate
# USD to EUR
USD_TO_EUR = 0.86


# Daily spending profiles
SPENDING_PROFILES = {

    "Economical": {
        "inexpensive_meals": 2,
        "midrange_meals": 0,
        "fast_food": 1,
        "coffee": 0,
        "soft_drinks": 1,
        "water": 2,
        "transport_tickets": 4,
        "taxi_rides": 0,
        "taxi_distance": 5,
        "entertainment": 0.25
    },

    "Standard": {
        "inexpensive_meals": 2,
        "midrange_meals": 1,
        "fast_food": 0,
        "coffee": 1,
        "soft_drinks": 1,
        "water": 2,
        "transport_tickets": 2,
        "taxi_rides": 1,
        "taxi_distance": 5,
        "entertainment": 0.5
    },

    "Premium": {
        "inexpensive_meals": 1,
        "midrange_meals": 2,
        "fast_food": 0,
        "coffee": 2,
        "soft_drinks": 2,
        "water": 2,
        "transport_tickets": 0,
        "taxi_rides": 3,
        "taxi_distance": 5,
        "entertainment": 1
    }
}


# City code mapping
CITY_MAPPING = {
    "ank": "ankara",
    "bak": "baku"
}


# Standardize city names
def normalize_city(value):

    value = unicodedata.normalize(
        "NFKC", str(value)
    )

    value = " ".join(
        value.strip().lower().split()
    )

    return CITY_MAPPING.get(value, value)


# Load daily expenses dataset
@st.cache_data
def load_daily_cost_data():

    df = pd.read_csv(DAILY_COST_PATH)

    df["city"] = df["city"].apply(
        normalize_city
    )

    df["country"] = (
        df["country"].str.strip().str.lower()
    )

    return df


# Find available countries for a city
def get_destination_countries(destination):

    df = load_daily_cost_data()

    city = normalize_city(destination)

    matches = df[df["city"] == city]

    return sorted(
        matches["country"].unique().tolist()
    )


# Calculate daily expenses
def predict_daily_expenses(user_input):

    df = load_daily_cost_data()

    destination = normalize_city(
        user_input["destination"]
    )

    country = user_input.get(
        "destination_country"
    )

    travelers = user_input["travelers"]

    days = user_input["trip_days"]

    style = user_input["spending_style"]

    profile = SPENDING_PROFILES[style]

    matches = df[df["city"] == destination]

    if country:

        matches = matches[
            matches["country"] ==
            country.strip().lower()
        ]

    if matches.empty:

        raise ValueError(
            f"No daily expense data for {destination}."
        )

    if len(matches) > 1:

        raise ValueError(
            f"Multiple countries found for {destination}. "
            "Please select a country."
        )

    prices = matches.iloc[0]

    # Retrieve required item prices
    # Missing prices must not be treated as zero
    def item(column, quantity):

        if quantity == 0:
            return 0.0

        price = prices[column]

        if pd.isna(price):

            raise ValueError(
                f"Missing price for {column} "
                f"in {destination}."
            )

        return float(price) * quantity


    # 1. Food per person per day
    food_daily = (

        item(
            "meal_inexpensive_usd",
            profile["inexpensive_meals"]
        )

        + item(
            "meal_midrange_two_usd",
            profile["midrange_meals"]
        ) / 2

        + item(
            "fast_food_usd",
            profile["fast_food"]
        )
    )


    # 2. Drinks per person per day
    drinks_daily = (

        item(
            "coffee_usd",
            profile["coffee"]
        )

        + item(
            "soft_drink_usd",
            profile["soft_drinks"]
        )

        + item(
            "water_usd",
            profile["water"]
        )
    )


    # 3. Public transportation
    transport_daily = item(
        "transport_ticket_usd",
        profile["transport_tickets"]
    )


    # 4. Shared taxi expenses
    number_of_taxis = math.ceil(
        travelers / 4
    )

    taxi_daily = (

        item(
            "taxi_start_usd",
            profile["taxi_rides"]
        )

        + item(
            "taxi_per_km_usd",
            profile["taxi_rides"]
            * profile["taxi_distance"]
        )

    ) * number_of_taxis


    # 5. Entertainment per person per day
    entertainment_daily = item(
        "cinema_usd",
        profile["entertainment"]
    )


    # Calculate total expenses in EUR
    food = (
        food_daily * days * travelers
        * USD_TO_EUR
    )

    drinks = (
        drinks_daily * days * travelers
        * USD_TO_EUR
    )

    transport = (
        transport_daily * days * travelers
        * USD_TO_EUR
    )

    taxi = (
        taxi_daily * days * USD_TO_EUR
    )

    entertainment = (
        entertainment_daily * days * travelers
        * USD_TO_EUR
    )


    # Round each expense category
    breakdown = {
        "food": round(food, 2),
        "drinks": round(drinks, 2),
        "transport": round(transport, 2),
        "taxi": round(taxi, 2),
        "entertainment": round(entertainment, 2)
    }

    total = sum(breakdown.values())

    return {

        **breakdown,

        "total": round(total, 2),

        "daily_per_person": round(
            total / (days * travelers), 2
        ),

        "total_per_person": round(
            total / travelers, 2
        ),

        "city": prices["city"],

        "country": prices["country"],

        "currency": "EUR",

        "exchange_rate": USD_TO_EUR,

        "spending_style": style
    }
