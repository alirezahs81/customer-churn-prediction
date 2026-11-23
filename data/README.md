# Dataset

Download the IBM Telco Customer Churn sample CSV from the repository root:

```bash
python -m src.download_data
```

The downloader verifies the documented SHA-256 before writing `telco_customer_churn.csv` here. Expected target: `Churn` (`Yes` / `No`). See [../DATASET.md](../DATASET.md) for source, license, attribution, and limitations.

Raw and generated files are ignored by Git; do not commit the dataset.
