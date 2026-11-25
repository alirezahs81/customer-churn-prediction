# Customer Churn Prediction

A reproducible, leakage-aware machine-learning project for customer churn classification using the IBM Telco Customer Churn sample dataset.

## Highlights

- Reproducible dataset download with SHA-256 verification.
- Stratified train/validation/test split (60/20/20).
- Preprocessing is fitted inside scikit-learn pipelines to prevent preprocessing leakage.
- Model selection and operating-threshold selection are based on validation data, not test metrics.
- The final model is refitted on the combined training and validation partitions before test evaluation.
- Automated tests use synthetic data and do not require the downloaded dataset.
- GitHub Actions runs lint and tests on Python 3.10, 3.11, and 3.12.

## Final evaluation snapshot

The following result is the approved evaluation snapshot for the IBM Telco Customer Churn dataset (7,043 rows), using Logistic Regression and the validation-selected operating threshold **0.63371099**.

| Test metric | Final value |
|---|---:|
| Accuracy | 0.7729 |
| Precision | 0.5592 |
| Recall | 0.6818 |
| F1 | 0.6145 |
| ROC-AUC | 0.8413 |

The minimum business Recall requirement was set to **0.65**. The validation result at the approved threshold was Precision 0.5920, Recall 0.6711, and F1 0.6291. On the reported test evaluation, Recall was 0.6818, exceeding the minimum requirement.

### Test confusion matrix

Rows are actual classes and columns are predicted classes; class 1 is Churn.

|  | Predicted: No Churn | Predicted: Churn |
|---|---:|---:|
| Actual: No Churn | 834 (TN) | 201 (FP) |
| Actual: Churn | 119 (FN) | 255 (TP) |

At this threshold, 255 of 374 customers who churned were identified, while 201 non-churning customers received a false churn alert. Precision of 0.5592 means approximately 56% of positive churn alerts were correct.

See [FINAL_EVALUATION.md](FINAL_EVALUATION.md) for the detailed metrics, operating-threshold rationale, and evaluation caveats.

> **Evaluation transparency:** the test partition was previously evaluated at an earlier validation-selected threshold (0.6098) and was subsequently re-evaluated at 0.63371099. The threshold choice itself was based on validation data, not test results, but the test partition was therefore not inspected only once. Treat these test metrics as a reported holdout snapshot rather than a pristine, never-before-inspected estimate. Do not use this test partition for further threshold or model tuning.

> **Implementation note:** the current `src.train` implementation still selects an F1-maximizing threshold on validation data. It does not yet enforce the project's separately agreed minimum Recall constraint of 0.65. Consequently, a fresh `python -m src.train` run may write a different threshold to `models/churn_threshold.json`; the approved threshold above is the one used for this documented final evaluation. Aligning the training and inference implementation with the Recall-constrained threshold policy is a follow-up code change.

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

The data is split using stratification and a fixed random seed. Model selection and threshold selection are performed on validation predictions. The final model is refitted on the combined training and validation partitions, then evaluated on the test partition. See the evaluation transparency note above for the actual history of test-partition inspection in this project run.

The source dataset is a cross-sectional sample, not a time-indexed customer history. This experiment therefore does not establish temporal forecasting performance, causal effects of retention actions, fairness across customer groups, or production readiness. Review [DATASET.md](DATASET.md) and [SECURITY.md](SECURITY.md) before using real customer data.

## License

The original code and documentation in this repository are released under the MIT License; see [LICENSE](LICENSE). The third-party IBM dataset is separately governed by the license and notices described in [DATASET.md](DATASET.md).
