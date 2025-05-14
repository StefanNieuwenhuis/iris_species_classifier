import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env into environment
env_file = ".env" if Path(".env").exists() else ".env.ci"
load_dotenv(dotenv_path=env_file)

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Env-based config
env_path = os.getenv("IRIS_DB_PATH")
if env_path is None:
    raise RuntimeError("Environment variable IRIS_DB_PATH is not set.")

RAW_DATA_PATH = BASE_DIR / env_path
IRIS_TABLE_NAME = os.getenv("IRIS_TABLE_NAME", "Iris")
SEED = int(os.getenv("SEED", 42))

# Derived paths
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA__DIR = DATA_DIR / "processed"
