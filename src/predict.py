"""Load a trusted trained pipeline and score a customer CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.pipeline import TARGET, prepare_features, validate_feature_frame

ROOT = Path(__file__).resolve().parents[1]


def predict(input_csv: Path, model_path: Path) -> pd.DataFrame:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the model first with `python -m src.train`."
        )
    customers = pd.read_csv(input_csv)
    if TARGET in customers.columns:
        customers = customers.drop(columns=[TARGET])
    customers = prepare_features(customers)
    validate_feature_frame(customers)

    # joblib uses pickle serialization: only load artifacts from a trusted training run.
    model = joblib.load(model_path)
    probability = model.predict_proba(customers)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return pd.DataFrame({"prediction": prediction, "churn_probability": probability})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "churn_pipeline.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "predictions.csv")
    args = parser.parse_args()
    result = predict(args.input, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
