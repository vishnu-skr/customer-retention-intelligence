from pathlib import Path

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "telco_churn.csv"

# Model Paths
MODELS_DIR = BASE_DIR / "models"
CHURN_MODEL_PATH = MODELS_DIR / "churn_model.pkl"

# App / Output Paths (optional future use)
OUTPUT_DIR = BASE_DIR / "outputs"
