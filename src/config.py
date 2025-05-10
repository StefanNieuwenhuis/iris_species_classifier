import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Env-based config
RAW_DATA_PATH = BASE_DIR / os.getenv("IRIS_DB_PATH")
IRIS_TABLE_NAME = os.getenv("IRIS_TABLE_NAME", "Iris")
SEED = int(os.getenv("SEED", 42))

# Derived paths
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA__DIR = DATA_DIR / "processed"
