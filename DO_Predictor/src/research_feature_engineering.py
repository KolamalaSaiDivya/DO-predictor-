"""Causal temporal feature engineering for research experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.research_config import LAG_STEPS, QUALITY_SUFFIX, ROLLING_WINDOWS, TIMESTAMP_COL
from src.logging_utils import setup_experiment_logger

logger = setup_experiment_logger(__name__)


def create_temporal_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    include_diff: bool = True,
    include_pct_change: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create lag, causal rolling, diff, and pct_change features.

    All rolling statistics use trailing windows only (center=False).
    Returns (feature_df, feature_dictionary).
    """
    lags = lags or LAG_STEPS
    rolling_windows = rolling_windows or ROLLING_WINDOWS
    base = df.copy()
    new_frames: list[pd.DataFrame] = []
    dictionary_rows: list[dict[str, object]] = []

    for col in columns:
        if col not in base.columns:
            continue
        col_frames: list[pd.DataFrame] = []
        for lag in lags:
            name = f"{col}_lag_{lag}"
            col_frames.append(pd.DataFrame({name: base[col].shift(lag)}, index=base.index))
            dictionary_rows.append(
                {
                    "feature_name": name,
                    "source_variable": col,
                    "feature_type": "lag",
                    "lag_window": lag,
                    "description": f"Causal lag {lag} of {col}",
                }
            )
        for window in rolling_windows:
            roll = base[col].rolling(window=window, min_periods=1)  # causal trailing window
            for stat, func in (
                ("rolling_mean", roll.mean),
                ("rolling_std", roll.std),
                ("rolling_min", roll.min),
                ("rolling_max", roll.max),
                ("rolling_median", roll.median),
            ):
                name = f"{col}_{stat}_{window}"
                col_frames.append(pd.DataFrame({name: func()}, index=base.index))
                dictionary_rows.append(
                    {
                        "feature_name": name,
                        "source_variable": col,
                        "feature_type": stat,
                        "lag_window": window,
                        "description": f"Causal {stat} over window {window} for {col}",
                    }
                )
        if include_diff:
            name = f"{col}_first_difference"
            col_frames.append(pd.DataFrame({name: base[col].diff()}, index=base.index))
            dictionary_rows.append(
                {
                    "feature_name": name,
                    "source_variable": col,
                    "feature_type": "first_difference",
                    "lag_window": 1,
                    "description": f"First difference of {col}",
                }
            )
        if include_pct_change:
            name = f"{col}_pct_change"
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = base[col].pct_change(fill_method=None)
            col_frames.append(pd.DataFrame({name: pct}, index=base.index))
            dictionary_rows.append(
                {
                    "feature_name": name,
                    "source_variable": col,
                    "feature_type": "percentage_change",
                    "lag_window": 1,
                    "description": f"Percentage change of {col}",
                }
            )
        if col_frames:
            new_frames.append(pd.concat(col_frames, axis=1))

    if new_frames:
        engineered = pd.concat(new_frames, axis=1)
        out = pd.concat([base, engineered], axis=1)
    else:
        out = base.copy()

    feature_dictionary = pd.DataFrame(dictionary_rows)
    logger.info("Created %d engineered features from %d source columns.", len(dictionary_rows), len(columns))
    return out, feature_dictionary


def add_forecast_target(
    df: pd.DataFrame,
    target_col: str,
    horizon_steps: int,
) -> pd.DataFrame:
    out = df.copy()
    out[f"{target_col}_target"] = out[target_col].shift(-horizon_steps)
    return out


def list_feature_columns(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    exclude_set = set(exclude)
    return [
        c
        for c in df.columns
        if c not in exclude_set
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.endswith(QUALITY_SUFFIX)
    ]
