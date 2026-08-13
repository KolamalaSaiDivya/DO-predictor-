"""Tests for RFE+CFS pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_selection import run_rfe_cfs


def test_rfe_cfs_pipeline():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({f"f{i}": rng.normal(size=300) for i in range(25)})
    y = pd.Series(X["f0"] * 3 + X["f2"] + rng.normal(scale=0.1, size=300))
    X_train, X_val = X.iloc[:200], X.iloc[200:]
    y_train, y_val = y.iloc[:200], y.iloc[200:]
    rfe, cfs, combined = run_rfe_cfs(X_train, y_train, X_val, y_val, feature_counts=[5, 10])
    assert len(combined.selected_features) <= len(rfe.selected_features)
