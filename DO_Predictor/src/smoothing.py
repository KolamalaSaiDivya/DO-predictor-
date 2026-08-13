"""Smoothed variants of each sensor series (moving average + Savitzky-Golay).

Added as extra `<col>_ma` / `<col>_sg` columns rather than overwriting the raw
signal, so the ablation study can compare "raw features" vs "smoothed features"
as a pipeline stage instead of losing the original data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.config import (
    MOVING_AVERAGE_WINDOW,
    SAVGOL_POLYORDER,
    SAVGOL_WINDOW,
    SENSOR_COLUMNS,
    get_logger,
)

logger = get_logger(__name__)


def moving_average_smooth(df: pd.DataFrame, window: int = MOVING_AVERAGE_WINDOW) -> pd.DataFrame:
    out = df.copy()
    for col in SENSOR_COLUMNS:
        if col not in out.columns:
            continue
        out[f"{col}_ma"] = out[col].rolling(window, min_periods=1, center=True).mean()
    return out


def savgol_smooth(df: pd.DataFrame, window: int = SAVGOL_WINDOW, polyorder: int = SAVGOL_POLYORDER) -> pd.DataFrame:
    """Savitzky-Golay needs a gap-free series, so we forward/back-fill a working
    copy just for the filter input - NaNs in the actual column are left alone.
    Short segments (< window) can't be filtered and are left as NaN.
    """
    out = df.copy()
    for col in SENSOR_COLUMNS:
        if col not in out.columns:
            continue
        working = out[col].ffill().bfill()
        if working.isna().all() or len(working) < window:
            out[f"{col}_sg"] = np.nan
            continue
        smoothed = savgol_filter(working.to_numpy(), window_length=window, polyorder=polyorder, mode="interp")
        result = pd.Series(smoothed, index=out.index)
        result[out[col].isna()] = np.nan  # don't pretend we smoothed data that was never there
        out[f"{col}_sg"] = result
    return out


def apply_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    out = moving_average_smooth(df)
    out = savgol_smooth(out)
    logger.info("Added moving-average and Savitzky-Golay columns for %d sensors.", len(SENSOR_COLUMNS))
    return out


if __name__ == "__main__":
    from src.cleaning import clean_pipeline
    from src.data_loader import load_raw_data
    from src.validation import validate_raw_data

    raw = load_raw_data()
    rpt = validate_raw_data(raw)
    clean_df, _ = clean_pipeline(raw, rpt.quality_columns_kept)
    smoothed = apply_smoothing(clean_df)
    print([c for c in smoothed.columns if c.endswith(("_ma", "_sg"))])
    print(smoothed[["Dissolved Oxygen", "Dissolved Oxygen_ma", "Dissolved Oxygen_sg"]].dropna().head())
