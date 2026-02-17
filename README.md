# Retail ML Decision Platform ☕

End-to-end machine learning system that forecasts product demand and converts predictions into operational decisions for retail store operations.

This project simulates a real production workflow similar to Starbucks store analytics: predicting daily demand, recommending staffing levels, and preparing inventory plans.

---

## Business Objective

Retail stores must decide daily:

- How many staff should work tomorrow?
- How much inventory should be prepared?
- Which products may run out?

This platform predicts future demand and automatically generates recommendations.

---

## System Architecture

Data → ML Model → API → Business Dashboard

Raw Sales Data
      ↓
ETL Pipeline (clean & transform)
      ↓
Feature Engineering (time series signals)
      ↓
Forecasting Model
      ↓
Prediction API (FastAPI)
      ↓
Decision Engine (staff & inventory)
      ↓
Manager Dashboard (Streamlit)

---

## Machine Learning

Forecasting target:
Daily units sold per store per product

Features:
- Lag demand (7 & 14 days)
- Rolling averages
- Day of week seasonality
- Monthly trends
- Weekend effects

Model:
Random Forest Regressor

Validation:
Time-based split (train on past → predict future)

Performance:
MAE ≈ 6.5 units

---

## Decision Intelligence Layer

Predictions are converted into operational actions:

Predicted Demand → Staff Needed
< 20 → 2 staff  
20–40 → 3 staff  
40–60 → 4 staff  
> 60 → 5 staff  

Inventory recommendation:
Prepare 20% buffer stock

---

## API Endpoints

Get prediction

GET /predict/{store_id}/{product_id}

Example response:

{
  "store_id": 1,
  "product_id": 1,
  "predicted_units_sold": 20.73
}

---

## Dashboard

Interactive interface for non-technical users to view:
- expected demand
- staffing recommendation
- inventory preparation

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- FastAPI
- Streamlit
- Time Series Feature Engineering
- REST API deployment pattern

---

## What This Demonstrates

- Applied machine learning
- Forecasting & feature engineering
- Model operationalization
- Business decision translation
- Communication to stakeholders

---

## How to Run Locally

pip install -r requirements.txt  
uvicorn api.app:app --reload  
streamlit run dashboard/streamlit_app.py

---
## Dataset

Download the dataset from Kaggle:

Store Item Demand Forecasting Challenge

Place `train.csv` inside:

data/raw/train.csv

Then run:

python -m src.pipelines.ingestion
python -m src.pipelines.feature_engineering

## Author
Juni

End-to-end data science system built as a production-style analytics platform project.

