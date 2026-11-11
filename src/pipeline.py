"""Reusable leakage-safe training and inference pipeline."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMNS = ("customerID", "CustomerID", "customer_id")
NUMERIC_CANDIDATES = ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen")


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic schema cleanup shared by training and inference."""
    features = frame.drop(columns=[c for c in ID_COLUMNS if c in frame.columns]).copy()
    for column in NUMERIC_CANDIDATES:
        if column in features.columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def load_dataset(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Dataset must contain target column {TARGET!r}.")
    target = frame[TARGET].astype("string").str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0}
    )
    frame = prepare_features(frame.drop(columns=[TARGET]))
    frame[TARGET] = target
    if frame[TARGET].isna().any():
        raise ValueError("Target column Churn must contain only Yes/No (or 1/0).")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Target column Churn must contain both classes.")
    if len(frame) < 10:
        raise ValueError("At least 10 rows are required for stratified splitting.")
    return frame


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = [
        column
        for column in X.columns
        if column in NUMERIC_CANDIDATES or pd.api.types.is_numeric_dtype(X[column])
    ]
    categorical = [column for column in X.columns if column not in numeric]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def build_estimators(random_state: int = 42) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }


def build_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("classifier", estimator),
        ]
    )


def validate_feature_frame(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Features must be supplied as a pandas DataFrame.")
    if X.empty or X.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    if TARGET in X.columns:
        raise ValueError("Do not include the target column in prediction features.")
