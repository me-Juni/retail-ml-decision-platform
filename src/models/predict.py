import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path("models_artifacts/demand_model.pkl")
MART_DATA = Path("data/mart/demand_features.csv")
OUTPUT_PATH = Path("data/mart/tomorrow_predictions.csv")

def load_model():
    model = joblib.load(MODEL_PATH)
    print("Model loaded")
    return model

def load_latest_data():
    df = pd.read_csv(MART_DATA, parse_dates=["date"])
    latest_date = df["date"].max()

    # simulate tomorrow prediction using latest available day
    latest_df = df[df["date"] == latest_date].copy()

    print("Predicting for next day after:", latest_date.date())
    return latest_df

def predict(model, df):

    features = [
        "store_id","product_id",
        "day_of_week","week_of_year","month","year",
        "is_weekend","lag_7","lag_14","rolling_mean_7"
    ]

    df["predicted_units_sold"] = model.predict(df[features])
    df["predicted_units_sold"] = df["predicted_units_sold"].round().astype(int)

    return df

def save_predictions(df):
    df.to_csv(OUTPUT_PATH, index=False)
    print("Predictions saved →", OUTPUT_PATH)

def run():
    model = load_model()
    latest_df = load_latest_data()
    preds = predict(model, latest_df)
    save_predictions(preds)

if __name__ == "__main__":
    run()
