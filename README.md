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

## Download data and train

The project uses the IBM Telco Customer Churn sample dataset. Download the source CSV and verify its SHA-256 before saving it locally:

```bash
python -m src.download_data
python -m src.eda
python -m src.train
```

See [DATASET.md](DATASET.md) for source, attribution, checksum, license information, and limitations. Raw data and generated artifacts are not committed to the repository.

## Project structure

- `.github/workflows/ci.yml`: continuous integration
- `src/`: dataset download, training, prediction, pipeline, and EDA code
- `tests/`: automated tests
- `data/`, `models/`, `outputs/`: local data and generated artifacts; excluded from Git

## Responsible use

This project is for educational and analytical use. Churn predictions are probabilistic estimates, not definitive judgments about individual customers. Review [SECURITY.md](SECURITY.md) before using real customer data.

## License

No license has been declared for this project's original code and documentation. The IBM dataset is a separate third-party work; see [DATASET.md](DATASET.md) and IBM's source repository license.
