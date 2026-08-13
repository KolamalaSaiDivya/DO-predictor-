"""Shared config: paths, column names, and hyperparameter defaults for the whole project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
TRAINED_MODELS_DIR: Final[Path] = PROJECT_ROOT / "trained_models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
EXPERIMENTS_DIR: Final[Path] = PROJECT_ROOT / "experiments"
RESULTS_DIR: Final[Path] = EXPERIMENTS_DIR / "results"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAINED_MODELS_DIR, REPORTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV_NAME: Final[str] = "brisbane_water_quality.csv"
RAW_CSV_PATH: Final[Path] = RAW_DATA_DIR / RAW_CSV_NAME
PROCESSED_CSV_PATH: Final[Path] = PROCESSED_DATA_DIR / "brisbane_water_quality_clean.parquet"

# --------------------------------------------------------------------------- #
# Kaggle dataset
# --------------------------------------------------------------------------- #
# id 5295659 in the brief resolves to this owner/slug on kaggle.
KAGGLE_DATASET_SLUG: Final[str] = "downshift/water-quality-monitoring-dataset"
KAGGLE_DATASET_FILE: Final[str] = "brisbane_water_quality.csv"

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
TIMESTAMP_COL: Final[str] = "Timestamp"
RECORD_NUMBER_COL: Final[str] = "Record number"

# pH is intentionally excluded - this is a Dissolved-Oxygen-only system, pH is
# dropped from the raw data at load time (see data_loader.py) and never enters
# the pipeline as a feature or a target.
SENSOR_COLUMNS: Final[list[str]] = [
    "Average Water Speed",
    "Average Water Direction",
    "Chlorophyll",
    "Temperature",
    "Dissolved Oxygen",
    "Dissolved Oxygen (%Saturation)",
    "Salinity",
    "Specific Conductance",
    "Turbidity",
]

# Water Speed/Direction have no paired quality column in the source file.
QUALITY_FLAGGED_COLUMNS: Final[list[str]] = [
    "Chlorophyll",
    "Temperature",
    "Dissolved Oxygen",
    "Dissolved Oxygen (%Saturation)",
    "Salinity",
    "Specific Conductance",
    "Turbidity",
]


def quality_col(base_col: str) -> str:
    return f"{base_col} [quality]"


TIME_CYCLICAL_COLUMNS: Final[list[str]] = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def get_exog_columns(target_col: str) -> list[str]:
    """Other raw sensor columns + cyclical time features for SARIMAX exog.

    A true seasonal SARIMAX term at daily period (144 steps @ 10min) is
    computationally infeasible here - state-space dim blows up and fitting
    doesn't finish in reasonable time. Cyclical hour/day-of-year features in
    exog give the model a cheap stand-in for that daily/annual seasonality.
    """
    return [c for c in SENSOR_COLUMNS if c != target_col] + TIME_CYCLICAL_COLUMNS


# No public code dictionary ships with this dataset. 1020 is the modal code
# (>90% of non-null flags across every column) - treated as "good", anything
# else non-null as "suspect". See validation.py.
QUALITY_GOOD_CODE: Final[int] = 1020

# --------------------------------------------------------------------------- #
# Target - Dissolved Oxygen only
# --------------------------------------------------------------------------- #
PRIMARY_TARGET: Final[str] = "Dissolved Oxygen"

# --------------------------------------------------------------------------- #
# Temporal structure
# --------------------------------------------------------------------------- #
# Dataset is described as "30-min interval" but ordering by Record number
# (the actually-monotonic field - raw Timestamp has 276 dupes + backward
# jumps) shows the modal step is 10 min, with a secondary 30-min mode and
# a couple of multi-day sensor-outage gaps. Resampling to a fixed 10-min
# grid so lag/rolling features and DL sequence windows mean the same thing
# everywhere.
RESAMPLE_FREQ: Final[str] = "10min"
MAX_INTERPOLATION_GAP: Final[int] = 6  # steps (60 min) - don't bridge gaps wider than this

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
LAG_STEPS: Final[list[int]] = [1, 2, 3, 6, 12, 48]  # 10,20,30,60,120,480 min
ROLLING_WINDOWS: Final[list[int]] = [6, 12, 48]  # 1h, 2h, 8h
FORECAST_HORIZON: Final[int] = 1  # t+1 step = 10 min ahead

# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #
SAVGOL_WINDOW: Final[int] = 11
SAVGOL_POLYORDER: Final[int] = 2
MOVING_AVERAGE_WINDOW: Final[int] = 5

# --------------------------------------------------------------------------- #
# Sequence models (DL)
# --------------------------------------------------------------------------- #
SEQUENCE_LENGTH: Final[int] = 24  # 4h of history at 10-min cadence

# --------------------------------------------------------------------------- #
# Train / val / test split - chronological, no shuffling
# --------------------------------------------------------------------------- #
TRAIN_FRACTION: Final[float] = 0.70
VAL_FRACTION: Final[float] = 0.15
TEST_FRACTION: Final[float] = 0.15

RANDOM_SEED: Final[int] = 42

# --------------------------------------------------------------------------- #
# Published baselines to beat (public Kaggle notebook, SARIMAX/Prophet on DO)
# --------------------------------------------------------------------------- #
BASELINE_SARIMAX_MAE: Final[float] = 0.60
BASELINE_PROPHET_MAE: Final[float] = 0.40

# --------------------------------------------------------------------------- #
# Low-DO monitoring threshold (flag only, no hardware action)
# --------------------------------------------------------------------------- #
# 4.0 mg/L is a standard hypoxia-risk threshold for aquatic ecosystems.
LOW_DO_THRESHOLD_MGL: Final[float] = 4.0

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
