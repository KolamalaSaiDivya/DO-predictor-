"""SARIMAX with the other sensor readings as exog - this is the model the cited
public-notebook baseline (MAE 0.60 on DO) used, so it needs to be a fair, faithful
implementation to compare against."""

from __future__ import annotations

from src.config import get_exog_columns
from src.models.base_model import StatsForecastAdapter


def build_sarimax(target_col: str) -> StatsForecastAdapter:
    # no seasonal_order term - a daily seasonal component at 144 steps (10-min
    # cadence) makes the state space blow up and fit never finishes in reasonable
    # time. cyclical hour/day-of-year columns in exog cover that instead.
    return StatsForecastAdapter(
        name="sarimax",
        order=(2, 1, 2),
        seasonal_order=None,
        exog_columns=get_exog_columns(target_col),
    )
