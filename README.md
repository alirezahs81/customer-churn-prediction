# Customer Churn Prediction

A reproducible machine-learning project for customer churn classification.

## Project status

- Preprocessing is implemented inside scikit-learn pipelines to prevent data leakage.
- Data is split into stratified train, validation, and test partitions.
- Validation F1 is used for model selection; the test set is held out for final evaluation.
- Baseline models: Logistic Regression, Random Forest, and Gradient Boosting.
- Automated tests use synthetic data and do not require private customer records.

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Run tests

```bash
python -m pytest -q
```

## Dataset

Place an authorized customer churn CSV at the path documented in [DATASET.md](DATASET.md). Dataset provenance and licensing must be verified before redistribution. No real customer data is committed to this repository.

## Project structure

- `.github/workflows/ci.yml`: continuous integration
- `src/`: training, prediction, pipeline, and EDA code
- `tests/`: automated tests
- `data/`, `models/`, `outputs/`: documentation placeholders; generated and private artifacts are excluded from Git

## Responsible use

This project is for educational and analytical use. Churn predictions are probabilistic estimates, not definitive judgments about individual customers. Review [SECURITY.md](SECURITY.md) before using real customer data.

## License

No license has been declared. All rights remain with their respective owners unless a license is added.
