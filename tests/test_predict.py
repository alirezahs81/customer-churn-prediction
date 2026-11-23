import json

import numpy as np
import pandas as pd
import pytest

from src.predict import predict


def test_predict_requires_model(tmp_path):
    path = tmp_path / "customers.csv"
    pd.DataFrame({"tenure": [10], "MonthlyCharges": [50.0]}).to_csv(path, index=False)
    with pytest.raises(FileNotFoundError, match="Train the model first"):
        predict(path, tmp_path / "missing.joblib")


def test_predict_rejects_target_column_to_prevent_leakage(tmp_path):
    path = tmp_path / "labeled_customers.csv"
    pd.DataFrame({"tenure": [10], "Churn": [1]}).to_csv(path, index=False)
    model_path = tmp_path / "trusted_model.joblib"
    model_path.touch()

    with pytest.raises(ValueError, match="must not contain target column"):
        predict(path, model_path)


def test_predict_rejects_empty_feature_frame(tmp_path):
    path = tmp_path / "empty_features.csv"
    pd.DataFrame({"customerID": ["C-001"]}).to_csv(path, index=False)
    model_path = tmp_path / "trusted_model.joblib"
    model_path.touch()

    with pytest.raises(ValueError, match="At least one feature column"):
        predict(path, model_path)


def test_predict_rejects_empty_customer_file(tmp_path, monkeypatch):
    from src import predict as predict_module

    class Model:
        feature_names_in_ = np.array(["tenure"])

    input_path = tmp_path / "customers.csv"
    pd.DataFrame({"tenure": pd.Series(dtype="int64")}).to_csv(input_path, index=False)
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    monkeypatch.setattr(predict_module.joblib, "load", lambda path: Model())

    with pytest.raises(ValueError, match="at least one customer row"):
        predict(input_path, model_path)


def test_predict_rejects_missing_required_feature(tmp_path, monkeypatch):
    from src import predict as predict_module

    class Model:
        feature_names_in_ = np.array(["tenure", "MonthlyCharges"])

    input_path = tmp_path / "customers.csv"
    pd.DataFrame({"tenure": [10]}).to_csv(input_path, index=False)
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    monkeypatch.setattr(predict_module.joblib, "load", lambda path: Model())

    with pytest.raises(ValueError, match="missing required columns"):
        predict(input_path, model_path)


def test_predict_rejects_unexpected_feature(tmp_path, monkeypatch):
    from src import predict as predict_module

    class Model:
        feature_names_in_ = np.array(["tenure"])

    input_path = tmp_path / "customers.csv"
    pd.DataFrame({"tenure": [10], "unexpected": [1]}).to_csv(input_path, index=False)
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    monkeypatch.setattr(predict_module.joblib, "load", lambda path: Model())

    with pytest.raises(ValueError, match="unexpected columns"):
        predict(input_path, model_path)


def test_predict_scores_customer_with_trained_pipeline(tmp_path):
    import joblib
    from src.pipeline import build_estimators, build_pipeline

    rng = np.random.default_rng(11)
    n = 80
    X = pd.DataFrame({
        "tenure": rng.integers(1, 72, n),
        "MonthlyCharges": rng.uniform(20, 120, n),
        "Contract": rng.choice(["Month-to-month", "One year"], n),
    })
    y = (X["tenure"] < 30).astype(int)
    model = build_pipeline(X, build_estimators()["Logistic Regression"])
    model.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    input_path = tmp_path / "customers.csv"
    X.iloc[:3].to_csv(input_path, index=False)

    result = predict(input_path, model_path)
    assert list(result.columns) == ["prediction", "churn_probability"]
    assert len(result) == 3
    assert result["churn_probability"].between(0, 1).all()


def test_predict_uses_saved_threshold_file(tmp_path, monkeypatch):
    class FixedProbabilityModel:
        def predict_proba(self, frame):
            return np.array([[0.4, 0.6], [0.2, 0.8]])

    input_path = tmp_path / "customers.csv"
    pd.DataFrame({"tenure": [1, 2]}).to_csv(input_path, index=False)
    model_path = tmp_path / "churn_pipeline.joblib"
    model_path.touch()
    threshold_path = tmp_path / "churn_threshold.json"
    threshold_path.write_text(json.dumps({"threshold": 0.7}), encoding="utf-8")
    monkeypatch.setattr("src.predict.joblib.load", lambda path: FixedProbabilityModel())

    result = predict(input_path, model_path, threshold_path)

    assert result["prediction"].tolist() == [0, 1]
