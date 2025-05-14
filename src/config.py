import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if it exists, else .env.ci
env_file = Path(".env") if Path(".env").exists() else Path(".env.ci")
load_dotenv(dotenv_path=env_file)

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent


def get_raw_data_path() -> Path:
    """
    Lazy-load IRIS_DB_PATH to compute its value at runtime.
    This allows patched environment variables to take effect.
    """
    env_path = os.getenv("IRIS_DB_PATH")
    if env_path is None:
        raise RuntimeError("Environment variable IRIS_DB_PATH is not set.")
    return BASE_DIR / env_path


IRIS_TABLE_NAME = os.getenv("IRIS_TABLE_NAME", "Iris")
SEED = int(os.getenv("SEED", 42))

DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA__DIR = DATA_DIR / "processed"
