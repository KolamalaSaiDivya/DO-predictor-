"""Lag/rolling/time features for the lag-matrix (sklearn-style) models, plus a
correlation matrix for the EDA notebook.

Sequence models (LSTM etc.) don't consume this output directly - they build
windows straight from the cleaned series in their adapters (see models/base_model.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    FORECAST_HORIZON,
    LAG_STEPS,
    ROLLING_WINDOWS,
    SENSOR_COLUMNS,
    TIMESTAMP_COL,
    get_logger,
)

logger = get_logger(__name__)


def add_lag_features(df: pd.DataFrame, columns: list[str] | None = None, lags: list[int] | None = None) -> pd.DataFrame:
    columns = columns or SENSOR_COLUMNS
    lags = lags or LAG_STEPS
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, columns: list[str] | None = None, windows: list[int] | None = None) -> pd.DataFrame:
    columns = columns or SENSOR_COLUMNS
    windows = windows or ROLLING_WINDOWS
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for w in windows:
            out[f"{col}_roll_mean{w}"] = out[col].rolling(w).mean()
            out[f"{col}_roll_std{w}"] = out[col].rolling(w).std()
    return out


def add_rate_of_change(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or SENSOR_COLUMNS
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[f"{col}_roc"] = out[col].diff()
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out[TIMESTAMP_COL]
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["day_of_year"] = ts.dt.dayofyear
    # cyclical encodings so midnight and 23:00 aren't treated as far apart
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    # Brisbane austral seasons: 12,1,2=summer 3,4,5=autumn 6,7,8=winter 9,10,11=spring
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    out["season"] = out["month"].map(season_map)
    return out


def add_target_shift(df: pd.DataFrame, target_col: str, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Add `<target>_target` = value `horizon` steps ahead - what every model predicts."""
    out = df.copy()
    out[f"{target_col}_target"] = out[target_col].shift(-horizon)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    horizon: int = FORECAST_HORIZON,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Full feature build for one target: lags, rolling stats, rate of change,
    time features, and the shifted target. No dropna here - the pipeline decides
    where to cut so the ablation stages stay comparable.
    """
    out = add_lag_features(df, lags=lags)
    out = add_rolling_features(out, windows=windows)
    out = add_rate_of_change(out)
    out = add_time_features(out)
    out = add_target_shift(out, target_col, horizon)
    logger.info("Built feature matrix: %d columns for target '%s'.", out.shape[1], target_col)
    return out


def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or SENSOR_COLUMNS
    present = [c for c in columns if c in df.columns]
    return df[present].corr()


if __name__ == "__main__":
    from src.cleaning import clean_pipeline
    from src.data_loader import load_raw_data
    from src.smoothing import apply_smoothing
    from src.validation import validate_raw_data

    raw = load_raw_data()
    rpt = validate_raw_data(raw)
    clean_df, _ = clean_pipeline(raw, rpt.quality_columns_kept)
    smoothed = apply_smoothing(clean_df)
    features = build_feature_matrix(smoothed, target_col="Dissolved Oxygen")
    print(features.shape)
    print(features.dropna().shape)
    print(correlation_matrix(clean_df).round(2))
