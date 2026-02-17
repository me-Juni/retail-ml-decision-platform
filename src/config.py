from pathlib import Path

# Root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data layers
RAW_DATA = BASE_DIR / "data" / "raw" / "train.csv"
STAGING_DATA = BASE_DIR / "data" / "staging" / "sales_clean.csv"