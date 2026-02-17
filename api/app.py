from fastapi import FastAPI
import pandas as pd
import joblib
from pathlib import Path
import os

# Import training pipeline
from src.models.train import run as train_model

app = FastAPI(title="Retail ML Decision Platform")

# Paths
MODEL_PATH = Path("models_artifacts/demand_model.pkl")
DATA_PATH = Path("data/mart/demand_features.csv")

FEATURES = [
    "store_id","product_id",
    "day_of_week","week_of_year","month","year",
    "is_weekend","lag_7","lag_14","rolling_mean_7"
]


# Ensure model exists (important for cloud deployment)
if not MODEL_PATH.exists():
    print("Model not found. Training model...")
    train_model()

model = joblib.load(MODEL_PATH)


# Routes
@app.get("/")
def home():
    return {"message": "Retail ML Decision Platform API running"}

@app.get("/predict/{store_id}/{product_id}")
def predict(store_id: int, product_id: int):

    # load feature dataset
    if not DATA_PATH.exists():
        return {"error": "Feature dataset not found. Run pipelines first."}

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # filter store/product
    subset = df[
        (df["store_id"] == store_id) &
        (df["product_id"] == product_id)
    ]

    if subset.empty:
        return {"error": "Store or product does not exist"}

    # use most recent available record
    row = subset.sort_values("date").tail(1)

    # predict
    prediction = model.predict(row[FEATURES])[0]

    return {
        "store_id": store_id,
        "product_id": product_id,
        "based_on_date": str(row["date"].iloc[0].date()),
        "predicted_units_sold": round(float(prediction), 2)
    }
