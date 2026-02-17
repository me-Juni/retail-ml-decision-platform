import pandas as pd
from src.config import STAGING_DATA
from pathlib import Path

MART_DATA = Path("data/mart/demand_features.csv")

def load_staging():
    df = pd.read_csv(STAGING_DATA, parse_dates=["date"])
    print("Loaded staging:", df.shape)
    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features"""
    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    print("Time features added")

    return df

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Forecasting features"""
    df = df.sort_values(["store_id", "product_id", "date"]).copy()

    df["lag_7"] = df.groupby(["store_id", "product_id"])["units_sold"].shift(7)
    df["lag_14"] = df.groupby(["store_id", "product_id"])["units_sold"].shift(14)

    df["rolling_mean_7"] = (
        df.groupby(["store_id", "product_id"])["units_sold"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    print("Lag features added")

    return df

def save_mart(df: pd.DataFrame):
    df = df.dropna()

    MART_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MART_DATA, index=False)

    print("Feature mart saved:", df.shape)
    print("Columns:", list(df.columns))

def run():
    df = load_staging()

    df = create_time_features(df)
    df = create_lag_features(df)

    save_mart(df)

if __name__ == "__main__":
    run()
