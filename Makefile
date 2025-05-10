.PHONY: lint test run format jupyter env

# Install dependencies in a virtual environment (assuming you are using `venv`)
env:
	python3 -m venv venv  # Create virtual environment
	. .venv/bin/activate && pip install -r requirements.txt

# Install Jupyter in the virtual environment (if using the venv for Jupyter)
jupyter:
	.venv/bin/jupyter lab notebooks/

# Linting code
lint:
	.venv/bin/ruff check .

# Format code with black
format:
	.venv/bin/black .

# Run the tests
test:
	.venv/bin/pytest

# Start the project (e.g., main script)
run:
	.venv/bin/python src/main.py
