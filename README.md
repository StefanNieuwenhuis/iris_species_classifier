# Iris Species Classifier

<a href="https://github.com/StefanNieuwenhuis/iris_species_classifier/actions/workflows/ci.yaml"><img src="https://github.com/StefanNieuwenhuis/iris_species_classifier/actions/workflows/ci.yaml/badge.svg" /></a>
<a href="https://codecov.io/gh/StefanNieuwenhuis/iris_species_classifier" ><img src="https://codecov.io/gh/StefanNieuwenhuis/iris_species_classifier/graph/badge.svg?token=94T80D5HC0"/></a>
<a href="https://github.com/psf/black/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://mypy-lang.org"><img src="https://www.mypy-lang.org/static/mypy_badge.svg" /></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>

A minimal implementation of a generative classifier using Gaussian Naive Bayes, trained and evaluated on the [classic Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). Built for clarity, reproducibility, and testing foundational machine learning concepts.

## Features

- Classifies Iris Flowers into three species using Gaussian Naive Bayes
- Includes test suite for core components (`/src/`)
- Code coverage via [pytest-cov](https://app.codecov.io/gh/StefanNieuwenhuis/iris_species_classifier)
- Static analysis with `black`, `mypy`, and `ruff`
- Makefile for all common tasks
- CI pipeline with GitHub Actions + Codecov integration

## Demo

Coming soon!

## Project structure

```bash
.
├── src/
|   ├── model/
|   |   └── gaussian_naive_bayes.py     # Core GaussianNB implementation
├── tests/                              # Unit tests
├── notebooks/
│   └──
├── Makefile                            # Automation: lint, test, format, etc.
├── requirements.txt                    # Runtime dependencies
├── requirements-ci.txt                 # CI dependencies
├── LICENSE.md                          # MIT LICENSE file
├── README.md                           # README with details
```

## Installation

```bash
git clone https://github.com/StefanNieuwenhuis/iris_species_classifier.git
cd iris_species_classifier

# Set up virtualenv
make env

# download data to /data/raw/
make download
```

## Usage

Run unit tests:

```bash
make test
```

Linting, Code Formatting, and Type checking:

```
make lint
make format
make typecheck
```

## Testing and Code Coverage

```bash
# Run unit tests
make test

# Run unit tests and generate code coverage report
make test-coverage
```

Outputs `coverage.xml` report. The CI uploads it to [pytest-cov](https://app.codecov.io/gh/StefanNieuwenhuis/iris_species_classifier).

## CI/CD

We use GitHub Actions to:
    - Lint, code format, and type checks on pull requests
    - Run unit tests and generate code coverage report
    - Upload code coverage to Codecov

## Author

**Stefan Nieuwenhuis**
Machine Learning Engineer focused on robust systems, end-to-end pipelines, and clarity in code.
[LinkedIn](https://www.linkedin.com/in/stefannhs) | [GitHub](https://github.com/StefanNieuwenhuis)

## License

This project is licensed under the MIT License – see [LICENSE](LICENSE.md) file for details.