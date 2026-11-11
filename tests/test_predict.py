import pandas as pd
import pytest
from src.predict import predict


def test_predict_requires_model(tmp_path):
    path = tmp_path / "customers.csv"
    pd.DataFrame({"tenure": [10], "MonthlyCharges": [50.0]}).to_csv(path, index=False)
    with pytest.raises(FileNotFoundError, match="Train the model first"):
        predict(path, tmp_path / "missing.joblib")


def test_predict_scores_customer_with_trained_pipeline(tmp_path):
    import joblib
    import numpy as np
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
