"""Tests ensuring no train/test leakage in feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_selection import run_rfe


def test_rfe_fit_on_train_only():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"f{i}": rng.normal(size=500) for i in range(10)})
    y = pd.Series(rng.normal(size=500))
    X_train, X_val = X.iloc[:400], X.iloc[400:]
    y_train, y_val = y.iloc[:400], y.iloc[400:]
    result = run_rfe(X_train, y_train, X_val, y_val, feature_counts=[3])
    assert all(f in X_train.columns for f in result.selected_features)
