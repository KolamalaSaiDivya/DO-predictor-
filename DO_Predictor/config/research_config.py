"""Central configuration for thesis research experiments (Chapter 3 pH, Chapter 5 DO)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
RESULTS_ROOT: Final[Path] = PROJECT_ROOT / "results"
CHAPTER3_RESULTS: Final[Path] = RESULTS_ROOT / "chapter3_ph"
CHAPTER5_RESULTS: Final[Path] = RESULTS_ROOT / "chapter5_do"
COMBINED_RESULTS: Final[Path] = RESULTS_ROOT / "combined"
MODELS_ROOT: Final[Path] = PROJECT_ROOT / "models"

for _path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, COMBINED_RESULTS, MODELS_ROOT):
    _path.mkdir(parents=True, exist_ok=True)

for _chapter in (CHAPTER3_RESULTS, CHAPTER5_RESULTS):
    for _sub in ("data", "models", "metrics", "predictions", "figures", "tables", "logs"):
        (_chapter / _sub).mkdir(parents=True, exist_ok=True)
(CHAPTER5_RESULTS / "feature_selection").mkdir(parents=True, exist_ok=True)
(MODELS_ROOT / "chapter3_ph").mkdir(parents=True, exist_ok=True)
(MODELS_ROOT / "chapter5_do").mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: Final[int] = 42

# --------------------------------------------------------------------------- #
# Dataset paths (independent experiments)
# --------------------------------------------------------------------------- #
CHAPTER3_DATA_PATH: Final[Path] = RAW_DATA_DIR / "chapter3_ph_dataset.csv"
CHAPTER5_DATA_PATH: Final[Path] = RAW_DATA_DIR / "chapter5_do_dataset.csv"

# Optional timestamp column overrides (None = auto-detect)
TIMESTAMP_COLUMN_CHAPTER3: str | None = None
TIMESTAMP_COLUMN_CHAPTER5: str | None = None

# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
CHAPTER3_TARGET: Final[str] = "pH"
CHAPTER5_TARGET: Final[str] = "DO"

# --------------------------------------------------------------------------- #
# Train / validation / test split (chronological, no shuffle)
# --------------------------------------------------------------------------- #
TRAIN_FRACTION: Final[float] = 0.70
VALIDATION_FRACTION: Final[float] = 0.15
TEST_FRACTION: Final[float] = 0.15

# Minimum non-null fraction required to keep an engineered feature in the matrix
MIN_FEATURE_COVERAGE: Final[float] = 0.85

# Chapter 3 optional 80/20 split for supplementary comparison
CHAPTER3_OPTIONAL_80_20: Final[bool] = True

# --------------------------------------------------------------------------- #
# Forecast horizon
# --------------------------------------------------------------------------- #
DO_FORECAST_MINUTES: Final[int] = 10  # 10-minute-ahead DO forecast
RESAMPLE_FREQ: Final[str] = "10min"  # regular grid for lag/rolling alignment

# --------------------------------------------------------------------------- #
# Temporal feature engineering
# --------------------------------------------------------------------------- #
LAG_STEPS: Final[list[int]] = [1, 2, 3, 5, 10, 15, 30]
ROLLING_WINDOWS: Final[list[int]] = [3, 5, 10, 15, 30]
LOOKBACK: Final[int] = 20  # sequence length for LSTM/GRU/Transformer

# --------------------------------------------------------------------------- #
# Deep learning hyperparameters
# --------------------------------------------------------------------------- #
LSTM_UNITS: Final[int] = 64
LSTM_DROPOUT: Final[float] = 0.2
LSTM_EPOCHS: Final[int] = 50
LSTM_BATCH_SIZE: Final[int] = 32
LSTM_LEARNING_RATE: Final[float] = 0.001
LSTM_PATIENCE: Final[int] = 8

# --------------------------------------------------------------------------- #
# ARIMA defaults (Chapter 3)
# --------------------------------------------------------------------------- #
ARIMA_P: int | None = 3
ARIMA_D: int | None = 1
ARIMA_Q: int | None = 2
ARIMA_AUTO_ORDER: Final[bool] = True  # search (p,d,q) on training data via AIC
ARIMA_MAX_P: Final[int] = 3
ARIMA_MAX_D: Final[int] = 2
ARIMA_MAX_Q: Final[int] = 3

# --------------------------------------------------------------------------- #
# Feature selection (Chapter 5)
# --------------------------------------------------------------------------- #
RFE_STEP: Final[int] = 1
RFE_FEATURE_COUNTS: Final[list[int]] = [5, 10, 15, 20, 25, 30]
CFS_MAX_FEATURES: int | None = 30  # None = use same counts as RFE sweep
FAST_RFE_STEP: Final[int] = 10

# --------------------------------------------------------------------------- #
# Low-DO alert thresholds (must be configured from study criterion)
# --------------------------------------------------------------------------- #
# 4.0 mg/L — same hypoxia-risk criterion as src.config.LOW_DO_THRESHOLD_MGL
LOW_DO_THRESHOLD: float | None = 4.0
WARNING_DO_THRESHOLD: float | None = None

# --------------------------------------------------------------------------- #
# Execution modes
# --------------------------------------------------------------------------- #
# FAST_MODE: smoke/integration testing only — NOT thesis results.
# FULL_MODE: complete experiments for thesis reporting.
FAST_MODE: bool = False
FULL_MODE: bool = True

# FAST_MODE overrides (applied when FAST_MODE=True) — NOT thesis results
FAST_MAX_ROWS: Final[int] = 1000
FAST_LSTM_EPOCHS: Final[int] = 2
FAST_ARIMA_AUTO: Final[bool] = False
FAST_SKIP_MODELS: Final[list[str]] = ["transformer", "gru", "svr", "lstm"]
FAST_LAG_STEPS: Final[list[int]] = [1, 2, 3]
FAST_ROLLING_WINDOWS: Final[list[int]] = [3, 5]
FAST_RFE_FEATURE_COUNTS: Final[list[int]] = [5, 10]
FAST_RFE_STEP: Final[int] = 5  # larger step = fewer RFE fits in smoke tests
FAST_CFS_MAX_FEATURES: Final[int] = 8
FAST_CHAPTER5_MODELS: Final[list[str]] = ["persistence", "linear_regression", "random_forest"]
FAST_CHAPTER5_FEATURE_SETS: Final[list[str]] = ["all_features", "rfe_cfs"]

# --------------------------------------------------------------------------- #
# Column aliases -> canonical names
# --------------------------------------------------------------------------- #
COLUMN_ALIASES: Final[dict[str, str]] = {
    "do": "DO",
    "dissolved oxygen": "DO",
    "dissolved_oxygen": "DO",
    "dissolved oxygen (%saturation)": "DO (%Saturation)",
    "dissolved oxygen (% saturation)": "DO (%Saturation)",
    "ph": "pH",
    "temperature": "Temperature",
    "temp": "Temperature",
    "salinity": "Salinity",
    "turbidity": "Turbidity",
    "chlorophyll": "Chlorophyll",
    "average water speed": "Average Water Speed",
    "water speed": "Average Water Speed",
    "average water direction": "Average Water Direction",
    "water direction": "Average Water Direction",
    "specific conductance": "Specific Conductance",
    "humidity": "Humidity",
    "timestamp": "Timestamp",
    "record number": "Record number",
}

# Brisbane dataset canonical mapping (Chapter 5 source file)
BRISBANE_COLUMN_MAP: Final[dict[str, str]] = {
    "Dissolved Oxygen": "DO",
    "Dissolved Oxygen (%Saturation)": "DO (%Saturation)",
}

QUALITY_SUFFIX: Final[str] = " [quality]"
QUALITY_GOOD_CODE: Final[int] = 1020
RECORD_NUMBER_COL: Final[str] = "Record number"
TIMESTAMP_COL: Final[str] = "Timestamp"

# Columns excluded from feature matrix
EXCLUDE_FROM_FEATURES: Final[list[str]] = [
    "Timestamp",
    "Record number",
    RECORD_NUMBER_COL,
]

# --------------------------------------------------------------------------- #
# Model sets per experiment
# --------------------------------------------------------------------------- #
CHAPTER3_MODELS: Final[list[str]] = [
    "persistence",
    "linear_regression",
    "random_forest",
    "svr",
    "arima",
    "lstm",
    "gru",
    "transformer",
    "arima_lstm_hybrid",
]

CHAPTER5_MODELS: Final[list[str]] = [
    "persistence",
    "linear_regression",
    "random_forest",
    "svr",
    "xgboost",
    "lstm",
    "gru",
    "transformer",
]

CHAPTER5_FEATURE_SETS: Final[list[str]] = ["all_features", "rfe", "cfs", "rfe_cfs"]

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def apply_mode(fast: bool = False, full: bool = False) -> None:
    """Set FAST_MODE / FULL_MODE at runtime (used by run_all.py)."""
    global FAST_MODE, FULL_MODE, LSTM_EPOCHS
    if fast:
        FAST_MODE = True
        FULL_MODE = False
        LSTM_EPOCHS = FAST_LSTM_EPOCHS
    elif full:
        FAST_MODE = False
        FULL_MODE = True
        LSTM_EPOCHS = 50


def effective_lstm_epochs() -> int:
    return FAST_LSTM_EPOCHS if FAST_MODE else LSTM_EPOCHS


def effective_max_rows() -> int | None:
    return FAST_MAX_ROWS if FAST_MODE else None


def effective_chapter3_models() -> list[str]:
    if FAST_MODE:
        return [m for m in CHAPTER3_MODELS if m not in FAST_SKIP_MODELS]
    return list(CHAPTER3_MODELS)


def effective_chapter5_models() -> list[str]:
    if FAST_MODE:
        return list(FAST_CHAPTER5_MODELS)
    return list(CHAPTER5_MODELS)


def effective_lag_steps() -> list[int]:
    return FAST_LAG_STEPS if FAST_MODE else LAG_STEPS


def effective_rolling_windows() -> list[int]:
    return FAST_ROLLING_WINDOWS if FAST_MODE else ROLLING_WINDOWS


def effective_rfe_feature_counts() -> list[int]:
    return FAST_RFE_FEATURE_COUNTS if FAST_MODE else RFE_FEATURE_COUNTS


def effective_cfs_max_features() -> int | None:
    return FAST_CFS_MAX_FEATURES if FAST_MODE else CFS_MAX_FEATURES


def effective_chapter5_feature_sets() -> list[str]:
    return FAST_CHAPTER5_FEATURE_SETS if FAST_MODE else CHAPTER5_FEATURE_SETS


def effective_rfe_step(n_features: int = 0) -> int:
    if FAST_MODE:
        return max(FAST_RFE_STEP, max(1, n_features // 15))
    return RFE_STEP


def set_python_hash_seed() -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))


LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
