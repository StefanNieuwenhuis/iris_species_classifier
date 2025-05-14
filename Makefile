.PHONY: download lint test run format format-check typecheck test-coverage jupyter env preflight

download:
	$(shell scripts/download_data.sh)

env:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

jupyter:
	.venv/bin/jupyter lab notebooks/

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/black .

format-check:
	.venv/bin/black --check .

typecheck:
	.venv/bin/mypy src tests

test:
	.venv/bin/pytest

test-coverage:
	.venv/bin/pytest --cov=src --cov-report=xml

preflight: lint format typecheck