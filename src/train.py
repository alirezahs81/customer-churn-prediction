"""Train, select on validation data, and evaluate once on an untouched test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.pipeline import TARGET, build_estimators, build_pipeline, load_dataset

ROOT = Path(__file__).resolve().parents[1]
MIN_CLASS_COUNT_FOR_SPLIT = 5


def classification_metrics(y_true, y_pred, y_probability) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
    }


def stratified_train_valid_test_split(X, y, random_state: int = 42):
    """Create a 60/20/20 split and fail clearly when either class is too rare."""
    counts = y.value_counts()
    if len(counts) != 2 or int(counts.min()) < MIN_CLASS_COUNT_FOR_SPLIT:
        raise ValueError(
            "A stratified 60/20/20 split requires at least "
            f"{MIN_CLASS_COUNT_FOR_SPLIT} rows in each Churn class; "
            f"observed counts: {counts.to_dict()}."
        )

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_valid,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def train(dataset_path: Path, output_dir: Path = ROOT) -> dict:
    data = load_dataset(str(dataset_path))
    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(int)

    X_train, X_valid, X_test, y_train, y_valid, y_test = (
        stratified_train_valid_test_split(X, y)
    )

    validation_results = {}
    for name, estimator in build_estimators().items():
        model = build_pipeline(X_train, estimator)
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_valid)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        validation_results[name] = classification_metrics(y_valid, prediction, probability)

    best_name = max(validation_results, key=lambda key: validation_results[key]["f1"])
    # After selection, refit the chosen estimator on train + validation. Test remains untouched.
    final_X = pd.concat([X_train, X_valid], axis=0)
    final_y = pd.concat([y_train, y_valid], axis=0)
    best_model = build_pipeline(final_X, build_estimators()[best_name])
    best_model.fit(final_X, final_y)
    test_probability = best_model.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= 0.5).astype(int)
    test_metrics = classification_metrics(y_test, test_prediction, test_probability)

    model_dir = output_dir / "models"
    outputs_dir = output_dir / "outputs"
    model_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / "churn_pipeline.joblib")

    report = {
        "protocol": {
            "split": "stratified 60/20/20 train/validation/test",
            "random_state": 42,
            "selection_metric": "validation F1 at threshold 0.5",
            "test_usage": "used once for final evaluation after model selection",
            "threshold": 0.5,
        },
        "best_model": best_name,
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
        "dataset_rows": int(len(data)),
        "positive_rate": float(y.mean()),
    }
    report_path = outputs_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "telco_customer_churn.csv",
        help="Path to the Telco customer churn CSV.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}. See DATASET.md for setup instructions."
        )
    print(json.dumps(train(args.data, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
