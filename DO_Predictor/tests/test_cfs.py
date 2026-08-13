"""Tests for CFS feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_selection import run_cfs


def test_cfs_returns_features():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({f"f{i}": rng.normal(size=100) for i in range(15)})
    y = pd.Series(X["f0"] + X["f1"] + rng.normal(scale=0.1, size=100))
    result = run_cfs(X, y, max_features=5)
    assert 1 <= len(result.selected_features) <= 5
