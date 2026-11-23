# Customer Churn Prediction

A reproducible, leakage-aware machine-learning project for customer churn classification using the IBM Telco Customer Churn sample dataset.

## Highlights

- Reproducible dataset download with SHA-256 verification.
- Stratified train/validation/test split (60/20/20).
- Preprocessing is fitted inside scikit-learn pipelines to prevent preprocessing leakage.
- Model and decision-threshold selection use validation data only; the test set is held out for final evaluation.
- The selected threshold is persisted and reused by the prediction CLI.
- Automated tests use synthetic data and do not require the downloaded dataset.
- GitHub Actions runs lint and tests on Python 3.10, 3.11, and 3.12.

## Results

The following snapshot was produced by a local training run on the documented IBM dataset. Results may vary if dependencies, data, or training code change.

| Test metric | Validation-selected threshold (0.6098) | Default threshold (0.5) |
|---|---:|---:|
| Accuracy | 0.7651 | 0.7381 |
| Precision | 0.5447 | 0.5043 |
| Recall | 0.7005 | 0.7834 |
| F1 | 0.6129 | 0.6136 |
| ROC-AUC | 0.8413 | 0.8413 |

The selected threshold trades some recall for higher precision and accuracy in this run; it does **not** improve F1. ROC-AUC is unchanged because it does not depend on a single classification threshold. Choose an operating threshold according to the relative cost of false positives and false negatives rather than treating one threshold as universally best.

## Requirements

- Python 3.10, 3.11, or 3.12
- pip
- Internet access for the initial dataset download

## Setup

### Windows PowerShell

From the repository directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

If PowerShell blocks virtual-environment activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Download data, explore, and train

Run from the repository root:

```bash
python -m src.download_data
python -m src.eda
python -m src.train
```

The downloader saves the verified CSV to `data/telco_customer_churn.csv`. See [DATASET.md](DATASET.md) for source, attribution, checksum, license, and dataset limitations. Raw data and generated artifacts are intentionally excluded from Git.

Training writes:

- `models/churn_pipeline.joblib`: fitted preprocessing and classifier pipeline
- `models/churn_threshold.json`: validation-selected classification threshold
- `outputs/evaluation_report.json`: validation and held-out test metrics

## Run predictions

After training, provide a CSV containing the same feature columns used during training. Do not include the target column `Churn`.

```bash
python -m src.predict --input path/to/customers.csv
```

Optional arguments:

- `--model`: path to a trusted trained model artifact
- `--threshold-file`: optional threshold JSON; by default, the CLI looks for `churn_threshold.json` beside the model
- `--output`: prediction CSV path (default: `outputs/predictions.csv`)

The prediction command checks required and unexpected feature columns against the trained model before scoring. It also rejects empty inputs and labeled input containing the target column.

**Security:** Joblib uses pickle serialization. Only load model artifacts from a trusted source; a malicious artifact can execute code during loading.

## Run tests and lint

```bash
python -m pytest -q
ruff check .
```

The CI workflow runs these checks on Python 3.10, 3.11, and 3.12.

## Project structure

- `.github/workflows/ci.yml`: continuous integration
- `src/download_data.py`: verified dataset download
- `src/eda.py`: exploratory analysis
- `src/pipeline.py`: preprocessing and model definitions
- `src/train.py`: data splitting, model selection, threshold selection, and evaluation
- `src/predict.py`: validated inference CLI
- `tests/`: automated tests
- `data/`, `models/`, `outputs/`: local data and generated artifacts (ignored by Git)

## Methodology and limitations

The data is split using stratification and a fixed random seed. Model selection and threshold selection are performed on validation predictions. The final model is refitted on the combined training and validation partitions, then evaluated on the untouched test partition.

The source dataset is a cross-sectional sample, not a time-indexed customer history. This experiment therefore does not establish temporal forecasting performance, causal effects of retention actions, fairness across customer groups, or production readiness. Review [DATASET.md](DATASET.md) and [SECURITY.md](SECURITY.md) before using real customer data.

## License

The original code and documentation in this repository are released under the MIT License; see [LICENSE](LICENSE). The third-party IBM dataset is separately governed by the license and notices described in [DATASET.md](DATASET.md).
