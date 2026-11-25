"""Train, select on validation data, and evaluate on a held-out test partition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
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
DEFAULT_THRESHOLD = 0.5
MINIMUM_RECALL = 0.65


def classification_metrics(y_true, y_pred, y_probability) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
    }


def select_threshold(
    y_true,
    y_probability,
    min_recall: float = MINIMUM_RECALL,
) -> tuple[float, dict[str, float]]:
    """Maximize validation precision while meeting the minimum Recall constraint."""
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError("min_recall must be between 0.0 and 1.0.")

    probabilities = np.asarray(y_probability, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    if probabilities.ndim != 1 or labels.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("y_true and y_probability must be one-dimensional arrays of equal length.")
    if len(labels) == 0:
        raise ValueError("At least one validation sample is required.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities must all be finite.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    # Match the fine-grained validation search: evaluate unique probabilities
    # rounded to eight decimal places, plus the conventional default threshold.
    candidates = np.unique(np.append(np.round(probabilities, 8), DEFAULT_THRESHOLD))
    eligible = []
    for threshold in candidates:
        prediction = (probabilities >= threshold).astype(int)
        metrics = classification_metrics(labels, prediction, probabilities)
        if metrics["recall"] >= min_recall:
            eligible.append((metrics, float(threshold)))

    if not eligible:
        raise ValueError(
            f"No threshold satisfies the minimum Recall requirement of {min_recall:.2f}."
        )

    # Maximize precision; deterministic tie-breaks prefer higher Recall,
    # then higher F1, then the higher threshold.
    best_metrics, best_threshold = max(
        eligible,
        key=lambda item: (
            item[0]["precision"],
            item[0]["recall"],
            item[0]["f1"],
            item[1],
        ),
    )
    return best_threshold, best_metrics


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
    validation_probabilities = {}
    for name, estimator in build_estimators().items():
        model = build_pipeline(X_train, estimator)
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_valid)[:, 1]
        prediction = (probability >= DEFAULT_THRESHOLD).astype(int)
        validation_probabilities[name] = probability
        validation_results[name] = classification_metrics(y_valid, prediction, probability)

    best_name = max(validation_results, key=lambda key: validation_results[key]["f1"])
    selected_threshold, tuned_validation_metrics = select_threshold(
        y_valid, validation_probabilities[best_name]
    )

    # Refit only after model and threshold selection. The test partition is not
    # used for model or threshold selection.
    final_X = pd.concat([X_train, X_valid], axis=0)
    final_y = pd.concat([y_train, y_valid], axis=0)
    best_model = build_pipeline(final_X, build_estimators()[best_name])
    best_model.fit(final_X, final_y)
    test_probability = best_model.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= selected_threshold).astype(int)
    test_metrics = classification_metrics(y_test, test_prediction, test_probability)
    test_default_prediction = (test_probability >= DEFAULT_THRESHOLD).astype(int)
    test_default_metrics = classification_metrics(
        y_test, test_default_prediction, test_probability
    )

    model_dir = output_dir / "models"
    outputs_dir = output_dir / "outputs"
    model_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / "churn_pipeline.joblib")
    threshold_path = model_dir / "churn_threshold.json"
    threshold_path.write_text(
        json.dumps(
            {
                "threshold": selected_threshold,
                "default_threshold": DEFAULT_THRESHOLD,
                "minimum_recall": MINIMUM_RECALL,
                "selection": "maximum validation precision subject to minimum validation Recall",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "protocol": {
            "split": "stratified 60/20/20 train/validation/test",
            "random_state": 42,
            "selection_metric": "validation F1 at threshold 0.5",
            "threshold_selection": (
                "maximum validation precision subject to validation Recall "
                f">= {MINIMUM_RECALL:.2f}"
            ),
            "minimum_recall": MINIMUM_RECALL,
            "test_usage": "excluded from model and threshold selection; used for final evaluation",
            "default_threshold": DEFAULT_THRESHOLD,
            "selected_threshold": selected_threshold,
        },
        "best_model": best_name,
        # validation_metrics uses threshold 0.5; test_metrics uses the selected
        # validation threshold for the final evaluation.
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
        "validation_metrics_at_default_threshold": validation_results,
        "validation_metrics_at_selected_threshold": tuned_validation_metrics,
        "test_metrics_at_selected_threshold": test_metrics,
        "test_metrics_at_default_threshold": test_default_metrics,
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
