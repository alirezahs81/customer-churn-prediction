"""Load a trusted trained pipeline and score a customer CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.pipeline import TARGET, prepare_features, validate_feature_frame

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLD = 0.5


def load_threshold(threshold_path: Path | None, model_path: Path) -> float:
    """Load the training-selected threshold, or use 0.5 for legacy model files."""
    path = threshold_path or model_path.with_name("churn_threshold.json")
    if not path.is_file():
        return DEFAULT_THRESHOLD

    payload = json.loads(path.read_text(encoding="utf-8"))
    threshold = float(payload["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold in {path} must be between 0 and 1.")
    return threshold


def validate_prediction_schema(customers: pd.DataFrame, model) -> None:
    """Reject ambiguous or incompatible inference data before calling the estimator."""
    if customers.columns.has_duplicates:
        duplicates = customers.columns[customers.columns.duplicated()].tolist()
        raise ValueError(f"Prediction input contains duplicate columns: {duplicates}.")
    if customers.empty:
        raise ValueError("Prediction input must contain at least one customer row.")

    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return

    expected_columns = set(expected)
    actual_columns = set(customers.columns)
    missing = sorted(expected_columns - actual_columns)
    unexpected = sorted(actual_columns - expected_columns)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing required columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        raise ValueError("Prediction input schema mismatch; " + "; ".join(details) + ".")


def predict(
    input_csv: Path,
    model_path: Path,
    threshold_path: Path | None = None,
) -> pd.DataFrame:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the model first with `python -m src.train`."
        )
    customers = pd.read_csv(input_csv)
    if TARGET in customers.columns:
        raise ValueError(
            f"Prediction input must not contain target column {TARGET!r}; "
            "remove labels before scoring to prevent target leakage."
        )
    customers = prepare_features(customers)
    validate_feature_frame(customers)

    # joblib uses pickle serialization: only load artifacts from a trusted training run.
    model = joblib.load(model_path)
    validate_prediction_schema(customers, model)
    probability = model.predict_proba(customers)[:, 1]
    threshold = load_threshold(threshold_path, model_path)
    prediction = (probability >= threshold).astype(int)
    return pd.DataFrame({"prediction": prediction, "churn_probability": probability})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "churn_pipeline.joblib")
    parser.add_argument(
        "--threshold-file",
        type=Path,
        default=None,
        help="Optional threshold JSON (defaults to churn_threshold.json beside the model).",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "predictions.csv")
    args = parser.parse_args()
    result = predict(args.input, args.model, args.threshold_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
