
import streamlit as st
import pandas as pd
from datetime import date, timedelta

# Page configuration
st.set_page_config(
    page_title="Travel Budget Predictor",
    page_icon="✈️",
    layout="wide"
)

from prediction import (
    predict_airfare,
    get_eligible_routes,
    predict_daily_expenses,
    get_destination_countries
)

# ==========================================
# 1. LOAD AVAILABLE ROUTES
# ==========================================

@st.cache_data
def load_supported_routes():

    flights = get_eligible_routes()

    destinations = flights[
        "destination_city"
    ].unique()

    supported_destinations = [
        city for city in destinations
        if get_destination_countries(city)
    ]

    flights = flights[
        flights["destination_city"].isin(
            supported_destinations
        )
    ]

    return flights


flights = load_supported_routes()

departure_cities = sorted(
    flights["origin_city"].unique().tolist()
)

# ==========================================
# 2. WEBSITE STYLING
# ==========================================

st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    .hero {
        background: linear-gradient(
            120deg, #103E46, #147D80
        );
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

    div[data-testid="stMetric"] {
        background-color: rgba(
            20, 125, 128, 0.08
        );
        padding: 20px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. WEBSITE HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <p>SMARTER TRAVEL PLANNING</p>
    <h1>Your next adventure starts here.</h1>
    <p>Predict future travel expenses and plan
    your personalized travel budget before
    you book.</p>
</div>
""", unsafe_allow_html=True)

st.header("Plan your trip")

st.write(
    "Enter your travel details below to "
    "estimate your future travel budget."
)

# ==========================================
# 4. DESTINATION SELECTION
# ==========================================

st.subheader("📍 Destination")

col1, col2 = st.columns(2)

with col1:

    origin = st.selectbox(
        "Departure city",
        options=departure_cities,
        index=None,
        placeholder="Select departure city"
    )

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

# Destination country matching
destination_country = None

if destination is not None:

    countries = get_destination_countries(
        destination
    )

    if len(countries) == 1:

        destination_country = countries[0]

        st.caption(
            f"Daily expense location: "
            f"{destination}, "
            f"{destination_country.title()}"
        )

    elif len(countries) > 1:

        destination_country = st.selectbox(
            "Destination country",
            options=countries,
            index=None,
            format_func=lambda x: x.title(),
            placeholder="Select destination country",
            key=f"country_{destination}"
        )

        st.caption(
            "Country selection determines local "
            "expense prices. Airfare country "
            "matching has not yet been verified."
        )

st.divider()

# ==========================================
# 5. USER INPUT FORM
# ==========================================

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

    st.caption(
        "Economical: affordable meals and public transport. "
        "Standard: balanced dining and transportation. "
        "Premium: more dining, taxis and entertainment."
    )

    submitted = st.form_submit_button(
        "Predict my travel budget",
        use_container_width=True
    )

# ==========================================
# 6. VALIDATE USER INPUT
# ==========================================

if submitted:

    if origin is None or destination is None:

        st.error(
            "Please select departure and arrival cities."
        )

    elif destination_country is None:

        st.error(
            "Please select the destination country."
        )

    elif return_date <= departure_date:

        st.error(
            "Return date must be after departure date."
        )

    elif rooms > travelers:

        st.error(
            "Number of rooms cannot exceed travelers."
        )

    else:

        nights = (return_date - departure_date).days

        trip_days = nights + 1

        days_until_departure = (
            departure_date - date.today()
        ).days

        user_input = {
            "origin": origin,
            "destination": destination,
            "destination_country": destination_country,
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

        try:

            # ==========================================
            # 7. GENERATE ESTIMATES
            # ==========================================

            with st.spinner(
                "Predicting your travel expenses..."
            ):

                airfare = predict_airfare(user_input)

                daily = predict_daily_expenses(user_input)

            subtotal = (
                airfare["total"] + daily["total"]
            )

            # ==========================================
            # 8. RESULTS OVERVIEW
            # ==========================================

            st.divider()

            st.header("Your travel budget estimate")

            st.subheader(
                f"{origin} → {destination}, "
                f"{destination_country.title()}"
            )

            st.caption(
                f"{departure_date.strftime('%d %b %Y')} "
                f"– {return_date.strftime('%d %b %Y')}"
            )

            col7, col8, col9, col10 = st.columns(4)

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

            col10.metric(
                "Spending style",
                spending_style
            )

            st.divider()

            st.metric(
                "Estimated budget subtotal",
                f"€{subtotal:,.2f}"
            )

            st.caption(
                "Includes airfare and other expenses. "
                "Accommodation is not included yet."
            )

            col11, col12, col13 = st.columns(3)

            col11.metric(
                "Airfare",
                f"€{airfare['total']:,.2f}"
            )

            col12.metric(
                "Accommodation",
                "Pending"
            )

            col13.metric(
                "Other expenses",
                f"€{daily['total']:,.2f}"
            )

            # ==========================================
            # 9. AIRFARE BREAKDOWN
            # ==========================================

            st.divider()

            st.subheader("✈️ Estimated airfare")

            col14, col15, col16 = st.columns(3)

            col14.metric(
                "Outbound / person",
                f"€{airfare['outbound']:,.2f}"
            )

            col15.metric(
                "Return / person",
                f"€{airfare['return']:,.2f}"
            )

            col16.metric(
                "Round trip / person",
                f"€{airfare['per_person']:,.2f}"
            )

            st.caption(
                "Predicted using our trained "
                "Recurrent Neural Network (RNN)."
            )

            # ==========================================
            # 10. OTHER EXPENSES BREAKDOWN
            # ==========================================

            st.divider()

            st.subheader("💳 Estimated other expenses")

            col17, col18, col19 = st.columns(3)

            col17.metric(
                "Total other expenses",
                f"€{daily['total']:,.2f}"
            )

            col18.metric(
                "Daily cost / person",
                f"€{daily['daily_per_person']:,.2f}"
            )

            col19.metric(
                "Trip cost / person",
                f"€{daily['total_per_person']:,.2f}"
            )

            st.markdown("#### Detailed expense breakdown")

            expenses = [
                ("🍽️ Food and dining", daily["food"]),
                ("☕ Drinks and refreshments", daily["drinks"]),
                ("🚌 Public transportation", daily["transport"]),
                ("🚕 Taxi expenses", daily["taxi"]),
                ("🎟️ Entertainment", daily["entertainment"])
            ]

            expense_table = []

            for category, amount in expenses:

                expense_table.append({
                    "Expense category": category,
                    "Daily per person (€)": round(
                        amount / (trip_days * travelers), 2
                    ),
                    "Total trip cost (€)": round(amount, 2)
                })

            expense_df = pd.DataFrame(expense_table)

            st.dataframe(
                expense_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Daily per person (€)":
                        st.column_config.NumberColumn(
                            format="€%.2f"
                        ),
                    "Total trip cost (€)":
                        st.column_config.NumberColumn(
                            format="€%.2f"
                        )
                }
            )

            st.caption(
                "Estimated using destination-specific "
                "prices and spending profiles."
            )

            # ==========================================
            # 11. EXPENSE VISUALIZATION
            # ==========================================

            st.divider()

            st.subheader("📊 Travel expense distribution")

            chart_data = pd.DataFrame({
                "Category": [
                    "Airfare",
                    "Food",
                    "Drinks",
                    "Public transport",
                    "Taxi",
                    "Entertainment"
                ],
                "Cost (€)": [
                    airfare["total"],
                    daily["food"],
                    daily["drinks"],
                    daily["transport"],
                    daily["taxi"],
                    daily["entertainment"]
                ]
            })

            st.bar_chart(
                chart_data,
                x="Category",
                y="Cost (€)",
                color="#147D80"
            )

            st.info(
                "Accommodation will be added to the "
                "total budget once its model is connected."
            )

            # ==========================================
            # 12. ESTIMATION LIMITATIONS
            # ==========================================

            with st.expander("How are expenses estimated?"):

                st.write(
                    "Airfare is predicted using a trained "
                    "Recurrent Neural Network and historical "
                    "monthly airfare sequences."
                )

                st.write(
                    "Other expenses are calculated using "
                    "destination-specific item prices, "
                    "trip duration, number of travelers "
                    "and predefined spending profiles."
                )

                st.write(
                    "Some missing local prices are estimated "
                    "using country-level median prices."
                )

                st.write(
                    f"Assumed currency conversion: "
                    f"1 USD = {daily['exchange_rate']} EUR."
                )

                st.write(
                    "Daily expense estimates do not currently "
                    "account for future inflation. "
                    "Entertainment is estimated using cinema "
                    "prices as a proxy."
                )

                st.write(
                    "All results are estimates, not guaranteed "
                    "booking prices. Forecast uncertainty "
                    "may increase for longer travel horizons."
                )

        except Exception as error:

            st.error(
                f"Unable to generate prediction: {error}"
            )

# ==========================================
# 13. FOOTER
# ==========================================

st.divider()

st.caption(
    "ML-Based Personalized Travel Budget Prediction "
    "| DS 340W | Penn State"
)
