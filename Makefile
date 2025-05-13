.PHONY: download env jupyter lint format format-check test run check

# Download and extract data (uses .env vars)
download:
	bash scripts/download_data.sh

# Create virtual environment and install dependencies
env:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# Launch Jupyter Lab in notebooks directory
jupyter:
	.venv/bin/jupyter lab notebooks/

# Lint code with ruff
lint:
	.venv/bin/ruff check .

# Format code with black
format:
	.venv/bin/black .

# Check formatting only
format-check:
	.venv/bin/black --check .

# Run unit tests
test:
	.venv/bin/pytest

# Run unit tests with coverage
test-coverage:
	.venv/bin/pytest --cov --cov-report xml:coverage.xml

# Run main app
run:
	.venv/bin/python src/main.py

