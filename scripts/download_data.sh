#!/bin/bash

set -e

echo "Downloading Iris dataset via Kaggle CLI..."

mkdir -p data/raw

# WARNING! This script assumes the presence of Kaggle CLI
kaggle datasets download -d uciml/iris -p data/raw/iris_species/
echo "✅ Download complete."

cd data/raw/iris_species && unzip -o iris.zip && rm iris.zip
echo "✅ Unzip complete."