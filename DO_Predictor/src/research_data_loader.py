"""Research-grade data loading with format support and column normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config.research_config import (
    BRISBANE_COLUMN_MAP,
    COLUMN_ALIASES,
    QUALITY_SUFFIX,
    RECORD_NUMBER_COL,
    TIMESTAMP_COL,
)
from src.logging_utils import setup_experiment_logger

logger = setup_experiment_logger(__name__)

TIMESTAMP_CANDIDATES = [
    "Timestamp",
    "timestamp",
    "DateTime",
    "datetime",
    "date_time",
    "Time",
    "time",
    "Date",
    "date",
]


@dataclass
class LoadReport:
    filename: str
    n_rows: int
    n_columns: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    duplicate_rows: int
    duplicate_timestamps: int
    time_range: tuple[str, str] | None
    sampling_interval_minutes: dict[str, float | None]
    column_mapping: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "missing_counts": self.missing_counts,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_timestamps": self.duplicate_timestamps,
            "time_range": self.time_range,
            "sampling_interval_minutes": self.sampling_interval_minutes,
            "column_mapping": self.column_mapping,
        }


def _read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format '{suffix}' for {path}")


def normalize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    mapping: dict[str, str] = {}
    renamed: dict[str, str] = {}
    for col in df.columns:
        stripped = str(col).strip()
        key = stripped.lower()
        canonical = COLUMN_ALIASES.get(key, stripped)
        if stripped in BRISBANE_COLUMN_MAP:
            canonical = BRISBANE_COLUMN_MAP[stripped]
        if canonical != stripped:
            mapping[stripped] = canonical
        renamed[stripped] = canonical
    out = df.rename(columns=renamed)
    return out, mapping


def detect_timestamp_column(df: pd.DataFrame, override: str | None = None) -> str:
    if override and override in df.columns:
        return override
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        sample = df[col].dropna().head(20)
        if sample.empty:
            continue
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.8:
                return col
        except (ValueError, TypeError):
            continue
    raise ValueError("Could not detect timestamp column. Set TIMESTAMP_COLUMN_CHAPTER3/5 in config.")


def compute_sampling_stats(df: pd.DataFrame, timestamp_col: str) -> dict[str, float | None]:
    ts = pd.to_datetime(df[timestamp_col], errors="coerce").sort_values()
    diffs = ts.diff().dropna().dt.total_seconds() / 60.0
    positive = diffs[diffs > 0]
    if positive.empty:
        return {"median": None, "mean": None, "mode": None}
    mode_val = float(positive.mode().iloc[0]) if not positive.mode().empty else None
    return {
        "median": float(positive.median()),
        "mean": float(positive.mean()),
        "mode": mode_val,
    }


def load_research_dataset(
    path: Path,
    timestamp_override: str | None = None,
    sort_chronologically: bool = True,
) -> tuple[pd.DataFrame, LoadReport]:
    """Load CSV/XLSX/Parquet, normalize columns, parse timestamps, sort."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the dataset file at this path or update config."
        )

    logger.info("Loading research dataset from %s", path)
    df = _read_file(path)
    df, column_mapping = normalize_column_names(df)

    timestamp_col = detect_timestamp_column(df, timestamp_override)
    if timestamp_col != TIMESTAMP_COL:
        df = df.rename(columns={timestamp_col: TIMESTAMP_COL})
        column_mapping[timestamp_col] = TIMESTAMP_COL

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    if df[TIMESTAMP_COL].isna().all():
        raise ValueError(f"Timestamp column '{TIMESTAMP_COL}' could not be parsed.")

    if sort_chronologically:
        if RECORD_NUMBER_COL in df.columns:
            df = df.sort_values(RECORD_NUMBER_COL)
        else:
            df = df.sort_values(TIMESTAMP_COL)
        df = df.reset_index(drop=True)

    sampling = compute_sampling_stats(df, TIMESTAMP_COL)
    time_min = df[TIMESTAMP_COL].min()
    time_max = df[TIMESTAMP_COL].max()
    time_range = (str(time_min), str(time_max)) if pd.notna(time_min) and pd.notna(time_max) else None

    report = LoadReport(
        filename=path.name,
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        columns=list(df.columns),
        dtypes={c: str(df[c].dtype) for c in df.columns},
        missing_counts={c: int(df[c].isna().sum()) for c in df.columns},
        duplicate_rows=int(df.duplicated().sum()),
        duplicate_timestamps=int(df[TIMESTAMP_COL].duplicated().sum()),
        time_range=time_range,
        sampling_interval_minutes=sampling,
        column_mapping=column_mapping,
    )

    logger.info(
        "Loaded %s: rows=%d cols=%d range=%s median_interval=%s min",
        path.name,
        report.n_rows,
        report.n_columns,
        time_range,
        sampling.get("median"),
    )
    logger.info("Columns: %s", report.columns)
    return df, report


def save_load_report(report: LoadReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")


def get_numeric_sensor_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Return valid numerical columns excluding timestamp, record id, quality codes."""
    exclude = {TIMESTAMP_COL, RECORD_NUMBER_COL, target_col, f"{target_col}{QUALITY_SUFFIX}"}
    cols: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.endswith(QUALITY_SUFFIX):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols
