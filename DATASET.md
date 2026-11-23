# Dataset provenance and redistribution

## Source and expected input

This project uses the IBM Telco Customer Churn sample dataset, which describes customers of a fictional telecommunications company.

- Publisher: IBM
- Source repository: https://github.com/IBM/telco-customer-churn-on-icp4d
- Direct CSV: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Repository license: Apache License 2.0 (see the source repository's `LICENSE`)
- Expected file: `data/telco_customer_churn.csv`
- Expected shape: 7,043 rows × 21 columns
- Target: `Churn`, with `Yes` / `No` values
- SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

The dataset file is downloaded by `src.download_data` and its SHA-256 is checked before it is saved. If the content differs from the documented snapshot, the downloader fails rather than silently accepting a changed file.

## Download and run

From the repository root:

```bash
python -m src.download_data
python -m src.eda
python -m src.train
```

The downloader saves the CSV locally at `data/telco_customer_churn.csv`. Raw data and generated artifacts are ignored by Git; do not commit the CSV or model/report outputs.

## Data limitations and responsible use

This is a cross-sectional sample, not a sequence of dated customer snapshots. The train/validation/test split is therefore stratified and random; it does not demonstrate temporal forecasting performance or causal impact of retention actions.

The dataset is a sample for educational/analytical use, not a production customer extract. Dropping `customerID` does not by itself establish anonymity or remove privacy obligations. Do not use this project to make consequential decisions about individuals without appropriate validation, oversight, and legal review.

## Attribution

If using the dataset, credit IBM and link to the original repository above. The Apache 2.0 license applies as stated in the source repository; review its terms and include required notices when redistributing material.
