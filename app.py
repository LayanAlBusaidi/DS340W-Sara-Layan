
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Travel Budget Predictor",
    page_icon="✈️",
    layout="wide"
)

# Load flight dataset
BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_flight_data():
    df = pd.read_csv(
        BASE_DIR / "data" / "fares_by_month_cleaned.csv"
    )

    df = df.dropna(
        subset=["origin_city", "destination_city"]
    )

    df["origin_city"] = df["origin_city"].str.strip()
    df["destination_city"] = df["destination_city"].str.strip()

    df = df[
        (df["origin_city"] != "") &
        (df["destination_city"] != "") &
        (df["origin_city"] != df["destination_city"])
    ]

    return df

flights = load_flight_data()

# Available departure cities
departure_cities = sorted(
    flights["origin_city"].unique().tolist()
)

# Website styling
st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    .hero {
        background: linear-gradient(120deg, #103E46, #147D80);
        padding: 40px;
        border-radius: 18px;
        color: white;
        margin-bottom: 30px;
    }

    .hero h1 {
        color: white;
        font-size: 38px;
    }

    .hero p {
        font-size: 17px;
        color: #D5E8E8;
    }

    div.stFormSubmitButton > button {
        background-color: #147D80;
        color: white;
        border-radius: 10px;
        min-height: 45px;
    }
</style>
""", unsafe_allow_html=True)

# Website header
st.markdown("""
<div class="hero">
    <p>SMARTER TRAVEL PLANNING</p>
    <h1>Your next adventure starts here.</h1>
    <p>Predict future travel expenses and plan your
    personalized travel budget before you book.</p>
</div>
""", unsafe_allow_html=True)

st.header("Plan your trip")

st.write(
    "Enter your travel details below to estimate "
    "your future travel budget."
)

# Destination selection
st.subheader("📍 Destination")

col1, col2 = st.columns(2)

with col1:
    origin = st.selectbox(
        "Departure city",
        options=departure_cities,
        index=None,
        placeholder="Select departure city"
    )

# Filter destinations based on selected origin
if origin is not None:
    arrival_cities = sorted(
        flights.loc[
            flights["origin_city"] == origin,
            "destination_city"
        ].unique().tolist()
    )
else:
    arrival_cities = []

with col2:
    destination = st.selectbox(
        "Arrival city",
        options=arrival_cities,
        index=None,
        placeholder="Select arrival city",
        disabled=origin is None,
        key=f"destination_{origin}"
    )

st.divider()

# User input form
with st.form("travel_form"):

    st.subheader("📅 Travel dates")

    col3, col4 = st.columns(2)

    with col3:
        departure_date = st.date_input(
            "Departure date",
            value=date.today() + timedelta(days=30),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=730)
        )

    with col4:
        return_date = st.date_input(
            "Return date",
            value=date.today() + timedelta(days=37),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=730)
        )

    st.divider()

    st.subheader("👥 Travelers and rooms")

    col5, col6 = st.columns(2)

    with col5:
        travelers = st.number_input(
            "Number of travelers",
            min_value=1,
            max_value=20,
            value=1,
            step=1
        )

    with col6:
        rooms = st.number_input(
            "Number of rooms",
            min_value=1,
            max_value=20,
            value=1,
            step=1
        )

    st.divider()

    st.subheader("💳 Spending preferences")

    spending_style = st.radio(
        "Travel spending style",
        ["Economical", "Standard", "Premium"],
        index=1,
        horizontal=True
    )

    submitted = st.form_submit_button(
        "Predict my travel budget",
        use_container_width=True
    )

# Process user inputs
if submitted:

    if origin is None or destination is None:
        st.error("Please select departure and arrival cities.")

    elif return_date <= departure_date:
        st.error("Return date must be after departure date.")

    elif rooms > travelers:
        st.error("Number of rooms cannot exceed travelers.")

    else:

        nights = (return_date - departure_date).days
        trip_days = nights + 1

        days_until_departure = (
            departure_date - date.today()
        ).days

        # Store user inputs for model integration
        user_input = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "departure_month": departure_date.strftime("%Y-%m"),
            "return_month": return_date.strftime("%Y-%m"),
            "travelers": travelers,
            "rooms": rooms,
            "nights": nights,
            "trip_days": trip_days,
            "days_until_departure": days_until_departure,
            "spending_style": spending_style
        }

        st.session_state["user_input"] = user_input

        st.success("Your travel details have been collected!")

        st.subheader("Your trip overview")

        st.write(f"**{origin} → {destination}**")

        col7, col8, col9 = st.columns(3)

        col7.metric(
            "Trip duration",
            f"{nights} nights"
        )

        col8.metric(
            "Travelers",
            travelers
        )

        col9.metric(
            "Days until departure",
            days_until_departure
        )

        st.info(
            "Travel details are ready. Cost predictions "
            "will appear once the models are connected."
        )

# Footer
st.divider()

col10, col11, col12 = st.columns(3)

with col10:
    st.markdown("### ✈️ Airfare")
    st.caption("Future flight price prediction")

with col11:
    st.markdown("### 🏨 Accommodation")
    st.caption("Hotel cost estimation")

with col12:
    st.markdown("### 💳 Other expenses")
    st.caption("Additional travel expense prediction")

st.divider()

st.caption(
    "ML-Based Personalized Travel Budget Prediction "
    "| DS 340W | Penn State"
)
