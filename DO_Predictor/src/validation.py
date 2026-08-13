"""Raw-data integrity checks: missingness, duplicates, timestamp order, impossible
values, and whether the [quality] columns are actually populated.

Read-only - this module reports, it doesn't touch the DataFrame. cleaning.py
does the actual fixing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    RECORD_NUMBER_COL,
    SENSOR_COLUMNS,
    TIMESTAMP_COL,
    get_logger,
    quality_col,
)

logger = get_logger(__name__)

# generous on purpose - these exist to catch sensor faults, not trim real variability
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "Average Water Speed": (0.0, 500.0),
    "Average Water Direction": (0.0, 360.0),
    "Chlorophyll": (0.0, 500.0),
    "Temperature": (-2.0, 45.0),
    "Dissolved Oxygen": (0.0, 25.0),
    "Dissolved Oxygen (%Saturation)": (0.0, 200.0),
    "Salinity": (0.0, 45.0),
    "Specific Conductance": (0.0, 100.0),
    "Turbidity": (0.0, 4000.0),
}


@dataclass
class QualityReport:
    n_rows: int
    n_columns: int
    missingness_pct: dict[str, float]
    duplicate_row_count: int
    duplicate_timestamp_count: int
    record_number_is_canonical_order: bool
    record_number_gap_count: int
    timestamp_monotonic_raw: bool
    timestamp_monotonic_by_record_order: bool
    timestamp_backward_jump_count: int
    modal_cadence_minutes: float | None
    secondary_cadence_minutes: float | None
    n_gaps_over_1h: int
    impossible_value_counts: dict[str, int]
    quality_columns_populated: dict[str, bool]
    quality_columns_dropped: list[str]
    quality_columns_kept: list[str]
    quality_good_fraction: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "missingness_pct": self.missingness_pct,
            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_timestamp_count": self.duplicate_timestamp_count,
            "record_number_is_canonical_order": self.record_number_is_canonical_order,
            "record_number_gap_count": self.record_number_gap_count,
            "timestamp_monotonic_raw": self.timestamp_monotonic_raw,
            "timestamp_monotonic_by_record_order": self.timestamp_monotonic_by_record_order,
            "timestamp_backward_jump_count": self.timestamp_backward_jump_count,
            "modal_cadence_minutes": self.modal_cadence_minutes,
            "secondary_cadence_minutes": self.secondary_cadence_minutes,
            "n_gaps_over_1h": self.n_gaps_over_1h,
            "impossible_value_counts": self.impossible_value_counts,
            "quality_columns_populated": self.quality_columns_populated,
            "quality_columns_dropped": self.quality_columns_dropped,
            "quality_columns_kept": self.quality_columns_kept,
            "quality_good_fraction": self.quality_good_fraction,
            "notes": self.notes,
        }


def _check_quality_columns(
    df: pd.DataFrame,
) -> tuple[dict[str, bool], list[str], list[str], dict[str, float]]:
    populated: dict[str, bool] = {}
    good_fraction: dict[str, float] = {}
    dropped: list[str] = []
    kept: list[str] = []

    for col in SENSOR_COLUMNS:
        qcol = quality_col(col)
        if qcol not in df.columns:
            continue
        non_null = df[qcol].notna().sum()
        is_populated = non_null > 0
        populated[qcol] = is_populated
        if is_populated:
            kept.append(qcol)
            good_fraction[qcol] = float((df[qcol] == 1020).sum() / non_null)
        else:
            dropped.append(qcol)
            good_fraction[qcol] = float("nan")

    return populated, dropped, kept, good_fraction


def _cadence_summary(df: pd.DataFrame) -> tuple[float | None, float | None, int, int]:
    """Step sizes ordered by Record number, not by re-sorting Timestamp (which
    would scramble rows sharing a timestamp)."""
    ordered = df.sort_values(RECORD_NUMBER_COL) if RECORD_NUMBER_COL in df.columns else df
    diffs = ordered[TIMESTAMP_COL].diff().dropna()
    diff_minutes = (diffs.dt.total_seconds() / 60.0).round(2)
    positive = diff_minutes[diff_minutes > 0]
    backward_jumps = int((diff_minutes < 0).sum())
    gaps_over_1h = int((diff_minutes > 60).sum())

    if positive.empty:
        return None, None, backward_jumps, gaps_over_1h

    counts = positive.value_counts()
    modal = float(counts.index[0])
    secondary = float(counts.index[1]) if len(counts) > 1 else None
    return modal, secondary, backward_jumps, gaps_over_1h


def validate_raw_data(df: pd.DataFrame) -> QualityReport:
    """Run all integrity checks on the raw frame and return a report."""
    notes: list[str] = []
    n_rows, n_columns = df.shape

    missingness_pct = (df.isna().mean() * 100).round(4).to_dict()

    duplicate_row_count = int(df.duplicated(subset=SENSOR_COLUMNS).sum())
    duplicate_timestamp_count = int(df[TIMESTAMP_COL].duplicated().sum())

    record_number_is_canonical = False
    record_number_gap_count = 0
    if RECORD_NUMBER_COL in df.columns:
        rn = df[RECORD_NUMBER_COL]
        record_number_is_canonical = bool(rn.is_monotonic_increasing and rn.is_unique)
        expected = np.arange(rn.iloc[0], rn.iloc[0] + len(rn))
        record_number_gap_count = int((rn.to_numpy() != expected).sum())
    else:
        notes.append(f"'{RECORD_NUMBER_COL}' column absent - can't verify canonical row order.")

    timestamp_monotonic_raw = bool(df[TIMESTAMP_COL].is_monotonic_increasing)
    ordered = df.sort_values(RECORD_NUMBER_COL) if RECORD_NUMBER_COL in df.columns else df
    timestamp_monotonic_by_record_order = bool(ordered[TIMESTAMP_COL].is_monotonic_increasing)

    modal_cadence, secondary_cadence, backward_jumps, gaps_over_1h = _cadence_summary(df)

    impossible_counts: dict[str, int] = {}
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        series = df[col]
        impossible_counts[col] = int(((series < low) | (series > high)).sum())

    populated, dropped, kept, good_fraction = _check_quality_columns(df)

    if not timestamp_monotonic_raw and timestamp_monotonic_by_record_order:
        notes.append(
            "Raw Timestamp isn't monotonic in file order but is monotonic once ordered by "
            f"'{RECORD_NUMBER_COL}' - Timestamp has clock jitter, Record number is the real "
            "acquisition order. Everything downstream orders by Record number."
        )
    if duplicate_timestamp_count > 0:
        notes.append(
            f"{duplicate_timestamp_count} duplicate Timestamp values (different Record numbers, "
            "same clock reading) - resolved in cleaning.py."
        )
    if backward_jumps > 0:
        notes.append(f"{backward_jumps} backward timestamp jump(s) even under Record-number order.")
    if modal_cadence is not None and modal_cadence != 30.0:
        notes.append(
            f"Modal sampling cadence is {modal_cadence:.0f} min, not the commonly-cited 30 min. "
            "RESAMPLE_FREQ in config.py is set to match."
        )
    if gaps_over_1h > 0:
        notes.append(f"{gaps_over_1h} gap(s) longer than 1 hour (sensor downtime).")
    if dropped:
        notes.append(f"Quality columns fully empty, dropped: {dropped}")
    if kept:
        notes.append(f"Quality columns populated with real codes, kept: {kept}")

    report = QualityReport(
        n_rows=n_rows,
        n_columns=n_columns,
        missingness_pct=missingness_pct,
        duplicate_row_count=duplicate_row_count,
        duplicate_timestamp_count=duplicate_timestamp_count,
        record_number_is_canonical_order=record_number_is_canonical,
        record_number_gap_count=record_number_gap_count,
        timestamp_monotonic_raw=timestamp_monotonic_raw,
        timestamp_monotonic_by_record_order=timestamp_monotonic_by_record_order,
        timestamp_backward_jump_count=backward_jumps,
        modal_cadence_minutes=modal_cadence,
        secondary_cadence_minutes=secondary_cadence,
        n_gaps_over_1h=gaps_over_1h,
        impossible_value_counts=impossible_counts,
        quality_columns_populated=populated,
        quality_columns_dropped=dropped,
        quality_columns_kept=kept,
        quality_good_fraction=good_fraction,
        notes=notes,
    )

    logger.info(
        "Validation done: %d rows, %d dup timestamps, %d quality cols kept, %d dropped, "
        "modal cadence=%s min.",
        n_rows,
        duplicate_timestamp_count,
        len(kept),
        len(dropped),
        modal_cadence,
    )
    for note in notes:
        logger.info("note: %s", note)

    return report


if __name__ == "__main__":
    import json

    from src.data_loader import load_raw_data

    raw = load_raw_data()
    rpt = validate_raw_data(raw)
    print(json.dumps(rpt.to_dict(), indent=2, default=str))
