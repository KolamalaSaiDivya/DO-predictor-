"""Plain ARIMA - no exogenous regressors, just the target's own history."""

from __future__ import annotations

from src.models.base_model import StatsForecastAdapter


def build_arima(target_col: str) -> StatsForecastAdapter:
    # (3,1,2) - decent general-purpose order for a noisy, slightly nonstationary sensor series
    return StatsForecastAdapter(name="arima", order=(3, 1, 2), exog_columns=None)
