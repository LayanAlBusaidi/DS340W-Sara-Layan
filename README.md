# Future Travel Budget Predictor

**DS 340W — Data Science Project**

## Project Overview

This project aims to develop a machine learning system that estimates personalized travel budgets based on destination, travel dates, trip duration, and user preferences.

Building on previous research in flight price prediction, hotel price forecasting, and AI-powered travel recommendations, our goal is to estimate multiple travel expenses and combine them into a total trip budget.

## Planned Features

- Flight price prediction using machine learning
- Hotel and accommodation cost estimation
- Estimation of additional travel expenses, including meals, transportation, and activities
- Personalized total travel budget calculation
- Comparison of travel costs across different destinations, dates, and trip durations
- Exploration of long-term forecasting accuracy
- Potential LLM integration to interpret user preferences and support personalized recommendations

## Datasets

### Flight Price Dataset

Source: https://github.com/Ayesh17/Flight_Price/blob/main/combined_data/combined_flight_data.csv

We plan to use the flight pricing dataset associated with our first parent paper to develop and evaluate airfare prediction models.

### Hotel Price Dataset

We have not yet selected a suitable hotel pricing dataset. We plan to identify historical hotel pricing data to support the accommodation forecasting component.

### Additional Travel Expenses

We also plan to incorporate estimates for meals, local transportation, activities, and other expenses. Data sources and estimation methods for these components are still being explored.

Our initial development will focus on flight price prediction before expanding toward a comprehensive travel budget estimator.

## Planned Models and Evaluation

We plan to compare the following machine learning models:

- Linear Regression
- Random Forest
- XGBoost

Model performance will be evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (R-squared)

We also plan to investigate the feasibility of integrating an LLM to support personalization and travel recommendations.

## Project Status

The project is currently in its initial development stage. Our next steps include preparing the flight pricing dataset, implementing baseline prediction models, identifying additional data sources, and developing the total budget estimation system.

## Team Members

- Layan Al Busaidi — lma5769
- Sara Al Riyami — ssa5596
