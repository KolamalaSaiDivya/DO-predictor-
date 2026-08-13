"""Tests for forecast alignment."""

from __future__ import annotations

import pandas as pd

from src.forecasting_utils import compute_horizon_steps


def test_ten_minute_horizon_on_regular_grid():
    ts = pd.date_range("2024-01-01", periods=100, freq="10min")
    df = pd.DataFrame({"Timestamp": ts, "DO": range(100)})
    report = compute_horizon_steps(df, forecast_minutes=10)
    assert report.horizon_steps == 1
