import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/mart/tomorrow_predictions.csv")
OUTPUT_PATH = Path("data/mart/store_decisions.csv")

def load_predictions():
    df = pd.read_csv(INPUT_PATH)
    return df

def staffing_rule(units):
    """simple operations rule"""
    if units < 20:
        return 2
    elif units < 40:
        return 3
    elif units < 60:
        return 4
    else:
        return 5

def inventory_rule(units):
    """prepare extra buffer stock"""
    return int(units * 1.2)

def create_decisions(df):

    df["recommended_staff"] = df["predicted_units_sold"].apply(staffing_rule)
    df["recommended_inventory"] = df["predicted_units_sold"].apply(inventory_rule)

    df["stock_risk"] = df["predicted_units_sold"] > 60

    return df

def save(df):
    df.to_csv(OUTPUT_PATH, index=False)
    print("Decision report generated →", OUTPUT_PATH)

def run():
    df = load_predictions()
    df = create_decisions(df)
    save(df)

if __name__ == "__main__":
    run()
