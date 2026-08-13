"""Tests for temporal feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from src.research_feature_engineering import create_temporal_features


def test_causal_rolling_uses_past_only():
    df = pd.DataFrame({"Temperature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out, _ = create_temporal_features(df, ["Temperature"], lags=[1], rolling_windows=[3])
    # first rolling mean at index 0 uses only current value (min_periods=1)
    assert out["Temperature_rolling_mean_3"].iloc[2] == pytest.approx(2.0)


def test_lag_features_shift_correctly():
    df = pd.DataFrame({"DO": [10.0, 20.0, 30.0]})
    out, _ = create_temporal_features(df, ["DO"], lags=[1], rolling_windows=[])
    assert pd.isna(out["DO_lag_1"].iloc[0])
    assert out["DO_lag_1"].iloc[1] == 10.0
