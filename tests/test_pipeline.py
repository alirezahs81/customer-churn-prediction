import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.pipeline import TARGET, build_estimators, build_pipeline, load_dataset
from src.train import stratified_train_valid_test_split, train


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


def test_training_writes_report_and_pipeline_artifact(tmp_path):
    path = tmp_path / "telco.csv"
    make_dataset(240).to_csv(path, index=False)
    report = train(path, tmp_path)
    assert report["protocol"]["split"].startswith("stratified 60/20/20")
    assert report["protocol"]["test_usage"].startswith("used once")
    assert set(report["validation_metrics"]) == set(build_estimators())
    assert set(report["test_metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert (tmp_path / "models" / "churn_pipeline.joblib").is_file()
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
