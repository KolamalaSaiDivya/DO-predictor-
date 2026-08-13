"""Download (or reuse) the Kaggle Brisbane buoy dataset and load it raw."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.config import (
    KAGGLE_DATASET_FILE,
    KAGGLE_DATASET_SLUG,
    RAW_CSV_PATH,
    RECORD_NUMBER_COL,
    TIMESTAMP_COL,
    get_logger,
)

logger = get_logger(__name__)

# this project is Dissolved-Oxygen-only - pH is dropped right at load time so
# it never reaches validation, cleaning, feature engineering, or any model
PH_COLUMNS_TO_DROP = ["pH", "pH [quality]"]


def download_dataset(force: bool = False) -> Path:
    """Pull the CSV via kagglehub into data/raw/, skipping if already cached."""
    if RAW_CSV_PATH.exists() and not force:
        logger.info("Raw CSV already present at %s - skipping download.", RAW_CSV_PATH)
        return RAW_CSV_PATH

    import kagglehub  # lazy import, only needed on first run

    logger.info("Downloading Kaggle dataset '%s' via kagglehub...", KAGGLE_DATASET_SLUG)
    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    source_csv = cache_dir / KAGGLE_DATASET_FILE
    if not source_csv.exists():
        candidates = list(cache_dir.rglob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV found in downloaded dataset cache at {cache_dir}")
        source_csv = candidates[0]
        logger.warning(
            "Expected file '%s' not found; using '%s' instead.", KAGGLE_DATASET_FILE, source_csv.name
        )

    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, RAW_CSV_PATH)
    logger.info("Copied raw dataset to %s (%.2f KB).", RAW_CSV_PATH, RAW_CSV_PATH.stat().st_size / 1024)
    return RAW_CSV_PATH


def load_raw_data(csv_path: Path | None = None, download_if_missing: bool = True) -> pd.DataFrame:
    """Load the raw CSV, row order untouched.

    Row order == Record number order (verified strictly increasing, no gaps/dupes),
    which is more trustworthy than raw Timestamp (has dupes + a couple backward jumps).
    """
    path = csv_path or RAW_CSV_PATH
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"Raw CSV not found at {path} and download_if_missing=False.")
        path = download_dataset()

    logger.info("Loading raw CSV from %s", path)
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])

    present = [c for c in PH_COLUMNS_TO_DROP if c in df.columns]
    if present:
        df = df.drop(columns=present)
        logger.info("Dropped pH column(s) %s - this system is Dissolved-Oxygen-only.", present)

    if RECORD_NUMBER_COL in df.columns and not df[RECORD_NUMBER_COL].is_monotonic_increasing:
        logger.warning(
            "'%s' is not monotonically increasing in this file - row order may not be chronological.",
            RECORD_NUMBER_COL,
        )

    logger.info("Loaded %d rows, %d columns.", df.shape[0], df.shape[1])
    return df


if __name__ == "__main__":
    frame = load_raw_data()
    print(frame.head())
    print(frame.shape)
