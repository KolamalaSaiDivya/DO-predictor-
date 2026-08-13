"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from src.evaluation import mean_absolute_percentage_error
from src.research_evaluation import compute_full_metrics, safe_mape_report


def test_mape_handles_zero():
    y = np.array([0.0, 1.0, 2.0])
    mape = mean_absolute_percentage_error(y, y)
    assert np.isfinite(mape)


def test_compute_full_metrics():
    y = np.array([1.0, 2.0, 3.0])
    metrics = compute_full_metrics(y, y)
    assert metrics["MAE"] == 0.0
    assert metrics["R2"] == 1.0


def test_safe_mape_report():
    report = safe_mape_report(np.array([0.0, 1.0]))
    assert "near_zero_excluded_count" in report
