import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

MART_DATA = Path("data/mart/demand_features.csv")
MODEL_PATH = Path("models_artifacts/demand_model.pkl")

def load_data():
    df = pd.read_csv(MART_DATA, parse_dates=["date"])
    return df

def time_split(df):
    """
    Train: before 2017
    Validation: 2017
    """
    train = df[df["date"] < "2017-01-01"]
    valid = df[df["date"] >= "2017-01-01"]

    return train, valid

def train_model(train, valid):

    features = [
        "store_id","product_id",
        "day_of_week","week_of_year","month","year",
        "is_weekend","lag_7","lag_14","rolling_mean_7"
    ]

    X_train = train[features]
    y_train = train["units_sold"]

    X_valid = valid[features]
    y_valid = valid["units_sold"]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)

    print(f"Validation MAE: {mae:.2f}")

    return model

def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model saved →", MODEL_PATH)

def run():
    df = load_data()
    train, valid = time_split(df)
    model = train_model(train, valid)
    save_model(model)

if __name__ == "__main__":
    run()
