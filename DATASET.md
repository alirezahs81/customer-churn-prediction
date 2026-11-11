# Dataset provenance and redistribution

## Expected input

Place the CSV at `data/telco_customer_churn.csv`. It must contain a `Churn` target column with `Yes`/`No` values and features compatible with the common Telco Customer Churn dataset.

## Provenance status

The supplied ZIP did not contain the raw CSV, authoritative source URL, publisher information, license statement, or redistribution permission. Provenance and license therefore remain **unverified**. Do not commit or redistribute raw, cleaned, or derived copies until the original source and terms have been checked. Record publisher, exact URL, license/version, attribution requirements, and redistribution restrictions here.

## Local setup

1. Obtain data from its authoritative source.
2. Review the license.
3. Save as `data/telco_customer_churn.csv`.
4. Run `python -m src.eda` and `python -m src.train`.

Raw data and generated artifacts are ignored by Git. Dropping `customerID` does not establish anonymity or remove privacy obligations.
