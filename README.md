# Iris Species Classifier

<a href="https://github.com/StefanNieuwenhuis/iris_species_classifier/actions/workflows/ci.yaml"><img src="https://github.com/StefanNieuwenhuis/iris_species_classifier/actions/workflows/ci.yaml/badge.svg" /></a>
<a href="https://codecov.io/gh/StefanNieuwenhuis/iris_species_classifier" ><img src="https://codecov.io/gh/StefanNieuwenhuis/iris_species_classifier/graph/badge.svg?token=94T80D5HC0"/></a>
<a href="https://github.com/psf/black/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://mypy-lang.org"><img src="https://www.mypy-lang.org/static/mypy_badge.svg" /></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>

![Iris Flower Species](./assets/iris_species_with_labels.png)

A minimal, educational implementation of classic and modern classifiers, trained and evaluated on the [classic Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). Built for clarity, reproducibility, and testing foundational machine learning concepts.

## Overview

This project implements, compares, and evaluates machine learning classifiers — starting with **Gaussian Naive Bayes from scratch**, and extending to **benchmark models** like K-Nearest Neighbors (KNN) and XGBoost. 

The focus is on:
- Understanding the math behind models  
- Building extensible ML pipelines  
- Enforcing clean, testable, and typed code  
- Comparing performance across implementations


## Models Implemented

| Model                | Status      |
|----------------------|-------------|
| Gaussian Naive Bayes | Implemented |
| Logistic Regression  | Planned     |
| K-Nearest Neighbors  | Planned     |
| ...                  |             |


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

## Jupyter Notebooks with JupyterLab

To start the JupyterLab environment and run notebooks:

```bash
make jupyter
```
Outputs `coverage.xml` report. The CI uploads it to [pytest-cov](https://app.codecov.io/gh/StefanNieuwenhuis/iris_species_classifier).

## CI/CD

We use GitHub Actions to:
    - Lint, code format, and type checks on pull requests
    - Run unit tests and generate code coverage report
    - Upload code coverage to Codecov

## Coming Soon

- Model comparison dashboards
- Docker + CI deployment
- Blog series on design decisions

## Author

**Stefan Nieuwenhuis**

Machine Learning Engineer specialized in Large-Scale Recommender Systems, 3+ Years Production ML, ex-Tumblr

[LinkedIn](https://www.linkedin.com/in/stefannhs) | [GitHub](https://github.com/StefanNieuwenhuis)

## License

This project is licensed under the MIT License – see [LICENSE](LICENSE.md) file for details.