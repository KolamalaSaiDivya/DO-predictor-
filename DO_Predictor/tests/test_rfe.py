"""Tests for RFE feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_selection import run_rfe


def test_rfe_selects_subset():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({f"f{i}": rng.normal(size=200) for i in range(20)})
    y = pd.Series(X["f0"] * 2 + rng.normal(scale=0.1, size=200))
    X_train, X_val = X.iloc[:150], X.iloc[150:]
    y_train, y_val = y.iloc[:150], y.iloc[150:]
    result = run_rfe(X_train, y_train, X_val, y_val, feature_counts=[5, 10])
    assert len(result.selected_features) <= 10
    assert len(result.selected_features) >= 1
