import pandas as pd
from src.config import RAW_DATA, STAGING_DATA

def load_raw_data() -> pd.DataFrame:
    """Load raw dataset"""
    df = pd.read_csv(RAW_DATA, parse_dates=["date"])
    print(f"Loaded raw data: {df.shape}")
    return df

def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dataset to company warehouse schema"""
    df = df.rename(columns={
        "store": "store_id",
        "item": "product_id",
        "sales": "units_sold"
    })

    df["store_id"] = df["store_id"].astype(int)
    df["product_id"] = df["product_id"].astype(int)
    df["units_sold"] = df["units_sold"].astype(int)

    return df

def save_staging(df: pd.DataFrame):
    STAGING_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(STAGING_DATA, index=False)
    print(f"Saved staging table → {STAGING_DATA}")

def run():
    raw = load_raw_data()
    clean = standardize_schema(raw)
    save_staging(clean)

if __name__ == "__main__":
    run()
