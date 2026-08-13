"""Turn the raw frame into a clean, uniformly-spaced time series.

Steps: null out sensor readings flagged bad by their [quality] column, resample
onto a fixed grid (this also resolves the duplicate-timestamp issue since dupes
just land in the same bin and get averaged), interpolate short gaps, flag
outliers (but don't drop them - that's a modeling decision, not a cleaning one).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    MAX_INTERPOLATION_GAP,
    QUALITY_GOOD_CODE,
    RECORD_NUMBER_COL,
    RESAMPLE_FREQ,
    SENSOR_COLUMNS,
    TIMESTAMP_COL,
    get_logger,
    quality_col,
)

logger = get_logger(__name__)

OUTLIER_ZSCORE_THRESHOLD = 4.0


def apply_quality_flags(df: pd.DataFrame, quality_cols: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Null out any reading whose paired quality code isn't the 'good' code.

    Only touches columns that actually have a populated quality column (passed
    in via quality_cols - validation.py already worked out which ones those are).
    Readings with no quality code at all are left as-is; we only distrust a
    value when the sensor explicitly flagged it.
    """
    out = df.copy()
    nulled_counts: dict[str, int] = {}

    for base_col in SENSOR_COLUMNS:
        qcol = quality_col(base_col)
        if qcol not in quality_cols or qcol not in out.columns:
            continue
        bad_mask = out[qcol].notna() & (out[qcol] != QUALITY_GOOD_CODE)
        nulled_counts[base_col] = int(bad_mask.sum())
        out.loc[bad_mask, base_col] = np.nan

    logger.info("Quality-flag nulling: %s", nulled_counts)
    return out, nulled_counts


def deduplicate_and_resample(df: pd.DataFrame, freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """Order by Record number, index by Timestamp, resample to a fixed grid.

    Averaging within each bin is what actually fixes the 280 duplicate-timestamp
    rows - they fall into the same bin and get merged.
    """
    ordered = df.sort_values(RECORD_NUMBER_COL) if RECORD_NUMBER_COL in df.columns else df.copy()
    indexed = ordered.set_index(TIMESTAMP_COL)

    numeric_cols = [c for c in SENSOR_COLUMNS if c in indexed.columns]
    resampled = indexed[numeric_cols].resample(freq).mean()
    counts = indexed[numeric_cols[0]].resample(freq).count().rename("n_source_records")

    result = resampled.join(counts).reset_index()
    logger.info(
        "Resampled %d raw rows -> %d bins at %s (from %d unique timestamps).",
        len(df), len(result), freq, df[TIMESTAMP_COL].nunique(),
    )
    return result


def interpolate_missing(df: pd.DataFrame, max_gap: int = MAX_INTERPOLATION_GAP) -> tuple[pd.DataFrame, dict[str, int]]:
    """Linear-interpolate gaps up to max_gap steps; longer gaps stay NaN.

    Real sensor outages (the two multi-day gaps found in validation) shouldn't be
    papered over with interpolation - leaving them NaN keeps that visible so
    feature_engineering/pipeline can decide to drop those rows.
    """
    out = df.copy()
    filled_counts: dict[str, int] = {}

    for col in SENSOR_COLUMNS:
        if col not in out.columns:
            continue
        before_na = out[col].isna().sum()
        out[col] = out[col].interpolate(method="linear", limit=max_gap, limit_area="inside")
        after_na = out[col].isna().sum()
        filled_counts[col] = int(before_na - after_na)

    logger.info("Interpolation filled: %s", filled_counts)
    return out, filled_counts


def flag_outliers(df: pd.DataFrame, z_threshold: float = OUTLIER_ZSCORE_THRESHOLD) -> pd.DataFrame:
    """Add a `<col>_outlier` boolean per sensor using a rolling z-score.

    A rolling (not global) mean/std because DO drifts seasonally over the
    ~10-month record - a fixed global threshold would flag half of winter as
    outliers relative to summer.
    """
    out = df.copy()
    window = 288  # ~2 days at 10-min cadence

    for col in SENSOR_COLUMNS:
        if col not in out.columns:
            continue
        roll_mean = out[col].rolling(window, min_periods=30, center=True).mean()
        roll_std = out[col].rolling(window, min_periods=30, center=True).std()
        z = (out[col] - roll_mean) / roll_std.replace(0, np.nan)
        out[f"{col}_outlier"] = (z.abs() > z_threshold).fillna(False)

    n_flagged = int(out[[f"{c}_outlier" for c in SENSOR_COLUMNS if f"{c}_outlier" in out.columns]].sum().sum())
    logger.info("Flagged %d outlier readings across all sensors (not dropped).", n_flagged)
    return out


def clean_pipeline(df: pd.DataFrame, quality_cols: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full cleaning sequence and return (clean_df, report)."""
    report: dict[str, Any] = {}

    flagged, nulled_counts = apply_quality_flags(df, quality_cols)
    report["quality_nulled_counts"] = nulled_counts

    resampled = deduplicate_and_resample(flagged)
    report["n_rows_after_resample"] = len(resampled)

    interpolated, filled_counts = interpolate_missing(resampled)
    report["interpolation_filled_counts"] = filled_counts

    final = flag_outliers(interpolated)
    remaining_na = final[SENSOR_COLUMNS].isna().sum().to_dict()
    report["remaining_na_after_cleaning"] = {k: int(v) for k, v in remaining_na.items()}

    logger.info("Cleaning pipeline complete: %d rows in final frame.", len(final))
    return final, report


if __name__ == "__main__":
    import json

    from src.data_loader import load_raw_data
    from src.validation import validate_raw_data

    raw = load_raw_data()
    rpt = validate_raw_data(raw)
    clean_df, clean_rpt = clean_pipeline(raw, rpt.quality_columns_kept)
    print(clean_df.head())
    print(json.dumps(clean_rpt, indent=2, default=str))
