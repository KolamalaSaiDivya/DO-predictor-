"""load -> validate -> clean -> smooth -> features -> train -> eval, wired together.

Stage dataframes (cleaned / smoothed / feature-engineered) are built once per
target and reused across every model in the registry - only the final feature
selection differs by "stage" (used for the preprocessing ablation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.cleaning import clean_pipeline
from src.config import (
    PROCESSED_CSV_PATH,
    RESAMPLE_FREQ,
    TEST_FRACTION,
    TIMESTAMP_COL,
    TRAIN_FRACTION,
    VAL_FRACTION,
    get_logger,
)
from src.data_loader import load_raw_data
from src.evaluation import StageMetricLogger
from src.feature_engineering import (
    add_lag_features,
    add_rate_of_change,
    add_rolling_features,
    add_target_shift,
    add_time_features,
    build_feature_matrix,
)
from src.models.base_model import BaseModel
from src.models.registry import SEQUENTIAL_MODELS, build_model
from src.smoothing import apply_smoothing
from src.validation import QualityReport, validate_raw_data

logger = get_logger(__name__)

STAGE_NAMES = ["cleaned", "smoothed", "lag_features", "full_features"]


@dataclass
class BaseArtifacts:
    """Shared, target-agnostic outputs of load -> validate -> clean -> smooth."""

    raw: pd.DataFrame
    quality_report: QualityReport
    cleaned: pd.DataFrame
    cleaning_report: dict[str, Any]
    smoothed: pd.DataFrame


def build_base_artifacts(csv_path: Path | None = None) -> BaseArtifacts:
    raw = load_raw_data(csv_path=csv_path) if csv_path else load_raw_data()
    quality_report = validate_raw_data(raw)
    cleaned, cleaning_report = clean_pipeline(raw, quality_report.quality_columns_kept)
    smoothed = apply_smoothing(cleaned)

    if csv_path is None:
        # cache the canonical cleaned+smoothed dataset - only for the bundled
        # dataset, so an uploaded/custom CSV never clobbers it
        smoothed.to_parquet(PROCESSED_CSV_PATH, index=False)
        logger.info("Cached processed dataset to %s", PROCESSED_CSV_PATH)

    return BaseArtifacts(raw, quality_report, cleaned, cleaning_report, smoothed)


def build_stage_dataframe(base: BaseArtifacts, target_col: str, stage: str) -> pd.DataFrame:
    """One dataframe per (target, stage) - later stages strictly add more
    processing so the comparison across stages isolates what each step buys."""
    if stage == "cleaned":
        out = add_lag_features(base.cleaned, lags=[1])
        out = add_target_shift(out, target_col)
    elif stage == "smoothed":
        out = add_lag_features(base.smoothed, lags=[1])
        out = add_target_shift(out, target_col)
    elif stage == "lag_features":
        out = add_lag_features(base.smoothed)
        out = add_rolling_features(out)
        out = add_rate_of_change(out)
        out = add_target_shift(out, target_col)
    elif stage == "full_features":
        out = build_feature_matrix(base.smoothed, target_col)
    else:
        raise ValueError(f"Unknown stage '{stage}'. Known stages: {STAGE_NAMES}")
    return out


def longest_contiguous_block(df: pd.DataFrame, freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """ARIMA/SARIMAX/Prophet (and later, the DL sequence models) need an
    unbroken timeline - a window/forecast spanning the 30-day sensor outage
    would mix unrelated periods. Finds the single longest gap-free run."""
    freq_minutes = pd.Timedelta(freq).total_seconds() / 60
    ts = df[TIMESTAMP_COL].reset_index(drop=True)
    diffs = ts.diff().dt.total_seconds() / 60
    diffs.iloc[0] = freq_minutes
    block_id = (diffs != freq_minutes).cumsum()
    best_block = block_id.value_counts().idxmax()
    mask = (block_id == best_block).to_numpy()
    return df.iloc[mask].reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRACTION,
    val_frac: float = VAL_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].reset_index(drop=True), df.iloc[train_end:val_end].reset_index(drop=True), df.iloc[val_end:].reset_index(drop=True)


def prepare_model_data(
    stage_df: pd.DataFrame, target_col: str, model_name: str
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    df = stage_df.dropna().reset_index(drop=True)
    if model_name in SEQUENTIAL_MODELS:
        df = longest_contiguous_block(df)

    train, val, test = chronological_split(df)
    target_label = f"{target_col}_target"

    splits: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for split_name, part in (("train", train), ("val", val), ("test", test)):
        X = part.drop(columns=[target_label])
        y = part[target_label]
        splits[split_name] = (X, y)
    return splits


MIN_TRAIN_ROWS = 100
MIN_TEST_ROWS = 20


def train_and_evaluate(
    model_name: str,
    target_col: str,
    stage_df: pd.DataFrame,
    stage_name: str,
    metric_logger: StageMetricLogger,
) -> dict[str, Any] | None:
    splits = prepare_model_data(stage_df, target_col, model_name)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    if len(X_train) < MIN_TRAIN_ROWS or len(X_test) < MIN_TEST_ROWS:
        logger.warning(
            "Skipping %s/%s/%s - not enough rows after dropna/contiguity (train=%d, test=%d).",
            stage_name, target_col, model_name, len(X_train), len(X_test),
        )
        return None

    model: BaseModel = build_model(model_name, target_col)
    logger.info(
        "Fitting %s on target=%s stage=%s (train=%d, val=%d, test=%d)...",
        model_name, target_col, stage_name, len(X_train), len(X_val), len(X_test),
    )
    model.fit(X_train, y_train, X_val, y_val)

    result: dict[str, Any] = {"model": model, "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)}
    for split_name, (X, y) in splits.items():
        metrics = model.evaluate(X, y)
        metric_logger.log(stage_name, target_col, model_name, split_name, metrics)
        result[f"{split_name}_metrics"] = metrics

    return result


if __name__ == "__main__":
    from src.config import PRIMARY_TARGET

    base = build_base_artifacts()
    stage_df = build_stage_dataframe(base, PRIMARY_TARGET, "full_features")
    ml = StageMetricLogger()
    res = train_and_evaluate("linear_regression", PRIMARY_TARGET, stage_df, "full_features", ml)
    print(res["test_metrics"])
