"""Print descriptive EDA without fitting preprocessing statistics or modifying source data."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline import TARGET

ROOT = Path(__file__).resolve().parents[1]


def summarize(path: Path) -> None:
    frame = pd.read_csv(path)
    if TARGET not in frame:
        raise ValueError(f"Dataset must contain target column {TARGET!r}.")
    if "TotalCharges" in frame:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    print(f"Shape: {frame.shape}")
    print("\nMissing values:\n", frame.isna().sum().sort_values(ascending=False))
    print("\nTarget distribution (%):\n", frame[TARGET].value_counts(normalize=True) * 100)
    print("\nNumeric summary:\n", frame.select_dtypes(include="number").describe().T)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "telco_customer_churn.csv")
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}. See DATASET.md.")
    summarize(args.data)


if __name__ == "__main__":
    main()
