"""Tests for validation-only classification threshold selection."""
import numpy as np

from src.train import select_threshold


def test_select_threshold_uses_validation_labels_and_returns_valid_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.10, 0.40, 0.35, 0.45, 0.60, 0.80])

    threshold, metrics = select_threshold(y_true, probabilities)

    assert 0.0 <= threshold <= 1.0
    assert metrics["f1"] >= 0.8
    assert metrics["recall"] >= 2 / 3
    assert metrics["roc_auc"] > 0.5


def test_select_threshold_is_deterministic_and_prefers_higher_threshold_on_ties():
    y_true = np.array([0, 1])
    probabilities = np.array([0.2, 0.8])

    first = select_threshold(y_true, probabilities)
    second = select_threshold(y_true, probabilities)

    assert first == second
    # Both 0.5 and 0.8 classify this tiny sample perfectly. The selector's
    # deterministic tie-break prefers the higher Recall, then higher threshold.
    assert first[0] == 0.8
    assert first[1]["precision"] == 1.0
    assert first[1]["recall"] == 1.0
