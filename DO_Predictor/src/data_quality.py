"""Data quality checks and preprocessing summaries for research experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.research_config import QUALITY_GOOD_CODE, QUALITY_SUFFIX, RECORD_NUMBER_COL, TIMESTAMP_COL
from src.logging_utils import setup_experiment_logger
from src.research_data_loader import get_numeric_sensor_columns

logger = setup_experiment_logger(__name__)


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing_count": missing,
                "missing_percentage": round(missing / len(df) * 100, 4) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def apply_quality_code_nulling(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    nulled: dict[str, int] = {}
    for col in numeric_cols:
        qcol = f"{col}{QUALITY_SUFFIX}"
        if qcol not in out.columns:
            continue
        bad = out[qcol].notna() & (out[qcol] != QUALITY_GOOD_CODE)
        nulled[col] = int(bad.sum())
        out.loc[bad, col] = np.nan
    return out, nulled


def deduplicate_timestamps(df: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
    ordered = df.sort_values(RECORD_NUMBER_COL) if RECORD_NUMBER_COL in df.columns else df.copy()
    numeric_cols = [c for c in ordered.columns if pd.api.types.is_numeric_dtype(ordered[c])]
    indexed = ordered.set_index(TIMESTAMP_COL)
    resampled = indexed[numeric_cols].resample(freq).mean().reset_index()
    return resampled


def build_preprocessing_summary(original: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in cleaned.columns:
        if not pd.api.types.is_numeric_dtype(cleaned[col]):
            continue
        orig_missing = int(original[col].isna().sum()) if col in original.columns else cleaned[col].isna().sum()
        final_missing = int(cleaned[col].isna().sum())
        series = cleaned[col].dropna()
        rows.append(
            {
                "column": col,
                "original_missing_count": orig_missing,
                "missing_percentage": round(orig_missing / len(original) * 100, 4) if len(original) else 0.0,
                "final_missing_count": final_missing,
                "min": float(series.min()) if not series.empty else np.nan,
                "max": float(series.max()) if not series.empty else np.nan,
                "mean": float(series.mean()) if not series.empty else np.nan,
                "std": float(series.std()) if not series.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def preprocess_dataset(
    df: pd.DataFrame,
    target_col: str,
    resample_freq: str = "10min",
    max_interpolation_gap: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Quality handling, resampling, short-gap interpolation, summary export."""
    numeric_cols = get_numeric_sensor_columns(df, target_col)
    cleaned, nulled = apply_quality_code_nulling(df, numeric_cols)
    cleaned = deduplicate_timestamps(cleaned, freq=resample_freq)

    filled_counts: dict[str, int] = {}
    for col in numeric_cols:
        if col not in cleaned.columns:
            continue
        before = cleaned[col].isna().sum()
        cleaned[col] = cleaned[col].interpolate(method="linear", limit=max_interpolation_gap, limit_area="inside")
        after = cleaned[col].isna().sum()
        filled_counts[col] = int(before - after)

    summary = build_preprocessing_summary(df, cleaned)
    report = {
        "quality_nulled": nulled,
        "interpolation_filled": filled_counts,
        "rows_before": len(df),
        "rows_after": len(cleaned),
    }
    logger.info("Preprocessing complete: %d -> %d rows", len(df), len(cleaned))
    return cleaned, summary, report
