"""Forecast horizon alignment and 10-minute-ahead target construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.research_config import DO_FORECAST_MINUTES, RESAMPLE_FREQ, TIMESTAMP_COL
from src.logging_utils import setup_experiment_logger

logger = setup_experiment_logger(__name__)


@dataclass
class HorizonReport:
    median_interval_minutes: float
    target_horizon_minutes: float
    horizon_steps: int
    resample_freq: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "median_interval_minutes": self.median_interval_minutes,
            "target_horizon_minutes": self.target_horizon_minutes,
            "horizon_steps": self.horizon_steps,
            "resample_freq": self.resample_freq,
        }


def infer_median_interval_minutes(df: pd.DataFrame) -> float:
    diffs = pd.to_datetime(df[TIMESTAMP_COL]).diff().dropna().dt.total_seconds() / 60.0
    positive = diffs[diffs > 0]
    if positive.empty:
        freq_minutes = pd.Timedelta(RESAMPLE_FREQ).total_seconds() / 60.0
        return float(freq_minutes)
    return float(positive.median())


def compute_horizon_steps(
    df: pd.DataFrame,
    forecast_minutes: int = DO_FORECAST_MINUTES,
    resample_freq: str = RESAMPLE_FREQ,
) -> HorizonReport:
    median_interval = infer_median_interval_minutes(df)
    freq_minutes = pd.Timedelta(resample_freq).total_seconds() / 60.0
    step_minutes = median_interval if np.isfinite(median_interval) and median_interval > 0 else freq_minutes
    horizon_steps = max(1, int(round(forecast_minutes / step_minutes)))
    actual_minutes = horizon_steps * step_minutes
    logger.info(
        "Forecast horizon: median_interval=%.2f min, steps=%d, actual=%.2f min",
        step_minutes,
        horizon_steps,
        actual_minutes,
    )
    return HorizonReport(
        median_interval_minutes=step_minutes,
        target_horizon_minutes=float(forecast_minutes),
        horizon_steps=horizon_steps,
        resample_freq=resample_freq,
    )


def align_predictions(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_steps: int,
) -> pd.DataFrame:
    n = min(len(timestamps), len(y_true), len(y_pred))
    ts = pd.to_datetime(timestamps.iloc[-n:]).reset_index(drop=True)
    target_ts = ts.shift(-horizon_steps)
    return pd.DataFrame(
        {
            "timestamp_input": ts,
            "timestamp_target": target_ts,
            "actual": y_true[-n:],
            "predicted": y_pred[-n:],
            "absolute_error": np.abs(y_true[-n:] - y_pred[-n:]),
            "squared_error": (y_true[-n:] - y_pred[-n:]) ** 2,
            "forecast_horizon_minutes": horizon_steps
            * (ts.diff().dt.total_seconds().median() / 60.0 if len(ts) > 1 else 0.0),
        }
    )
