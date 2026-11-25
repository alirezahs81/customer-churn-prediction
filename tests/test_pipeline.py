import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.pipeline import TARGET, build_estimators, build_pipeline, load_dataset
from src.train import (
    MINIMUM_RECALL,
    select_threshold,
    stratified_train_valid_test_split,
    train,
)


def make_dataset(n=160):
    rng = np.random.default_rng(7)
    tenure = rng.integers(1, 72, n)
    monthly = rng.uniform(20, 120, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    probability = 1 / (1 + np.exp((tenure - 30) / 9))
    churn = np.where(rng.random(n) < probability, "Yes", "No")
    return pd.DataFrame({
        "customerID": [f"C{i}" for i in range(n)],
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,
        "Contract": contract,
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "Churn": churn,
    })


def test_load_dataset_drops_identifier_and_encodes_target(tmp_path):
    path = tmp_path / "sample.csv"
    make_dataset().to_csv(path, index=False)
    data = load_dataset(str(path))
    assert "customerID" not in data and set(data[TARGET].unique()) == {0, 1}


def test_preprocessing_is_inside_pipeline_and_handles_unseen_categories():
    data = make_dataset()
    X = data.drop(columns=TARGET)
    y = data[TARGET].map({"Yes": 1, "No": 0})
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_pipeline(X_train, build_estimators()["Logistic Regression"])
    model.fit(X_train, y_train)
    unseen = X_test.iloc[[0]].copy()
    unseen["Contract"] = "New contract category"
    assert model.predict_proba(unseen).shape == (1, 2)


def test_select_threshold_maximizes_precision_with_recall_constraint():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.95, 0.85, 0.75, 0.25, 0.90, 0.70, 0.30, 0.10])

    threshold, metrics = select_threshold(y_true, probabilities, min_recall=0.65)

    assert threshold == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.75)


def test_select_threshold_rejects_invalid_minimum_recall():
    y_true = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8])

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        select_threshold(y_true, probabilities, min_recall=1.1)


def test_training_writes_report_and_pipeline_artifact(tmp_path):
    path = tmp_path / "telco.csv"
    make_dataset(240).to_csv(path, index=False)
    report = train(path, tmp_path)
    assert report["protocol"]["split"].startswith("stratified 60/20/20")
    assert "excluded from model and threshold selection" in report["protocol"]["test_usage"]
    assert report["protocol"]["minimum_recall"] == MINIMUM_RECALL
    assert report["validation_metrics_at_selected_threshold"]["recall"] >= MINIMUM_RECALL
    assert set(report["validation_metrics"]) == set(build_estimators())
    assert set(report["test_metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert (tmp_path / "models" / "churn_pipeline.joblib").is_file()
    threshold_path = tmp_path / "models" / "churn_threshold.json"
    assert threshold_path.is_file()
    saved_threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    assert saved_threshold["threshold"] == report["protocol"]["selected_threshold"]
    assert saved_threshold["minimum_recall"] == MINIMUM_RECALL
    assert "maximum validation precision" in saved_threshold["selection"]
    assert (tmp_path / "outputs" / "evaluation_report.json").is_file()


def test_invalid_target_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    frame = make_dataset()
    frame["Churn"] = "maybe"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Yes/No"):
        load_dataset(str(path))


def test_total_charges_blank_strings_are_coerced_to_missing(tmp_path):
    path = tmp_path / "telco_blanks.csv"
    frame = make_dataset()
    frame["TotalCharges"] = frame["TotalCharges"].astype(str)
    frame.loc[0, "TotalCharges"] = " "
    frame.to_csv(path, index=False)
    loaded = load_dataset(str(path))
    assert pd.isna(loaded.loc[0, "TotalCharges"])


def test_stratified_split_rejects_too_few_minority_examples():
    X = pd.DataFrame({"feature": range(20)})
    y = pd.Series([0] * 16 + [1] * 4)
    with pytest.raises(ValueError, match="at least 5 rows in each Churn class"):
        stratified_train_valid_test_split(X, y)
