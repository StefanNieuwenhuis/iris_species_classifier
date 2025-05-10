#!/bin/bash
set -e

# Load environment variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found!"
  exit 1
fi

# Check if the database already exists
if [ -f "$IRIS_DB_PATH" ]; then
    echo "Database already exists at $IRIS_DB_PATH — skipping download."
    exit 0
fi

echo "Downloading Iris dataset to $IRIS_DATA_DIR"

mkdir -p data/raw

# WARNING! This script assumes the presence of Kaggle CLI
kaggle datasets download -d uciml/iris -p "$IRIS_DATA_DIR"
echo "✅ Download complete."

cd "$IRIS_DATA_DIR" && unzip -o iris.zip && rm iris.zip
echo "✅ Unzip complete."