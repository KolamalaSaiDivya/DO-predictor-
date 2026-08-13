"""Extended evaluation helpers with timing and ranking."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.evaluation import compute_metrics, mean_absolute_percentage_error


def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    base = compute_metrics(y_true, y_pred)
    mse = float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    return {
        "MAE": base["mae"],
        "MSE": mse,
        "RMSE": base["rmse"],
        "MAPE": base["mape"],
        "R2": base["r2"],
        "n": base["n"],
    }


@contextmanager
def timer() -> Iterator[list[float]]:
    start = time.perf_counter()
    container = [0.0]
    try:
        yield container
    finally:
        container[0] = time.perf_counter() - start


def rank_models(comparison_df: pd.DataFrame) -> pd.DataFrame:
    ranked = comparison_df.copy()
    ranked["rank_rmse"] = ranked["RMSE"].rank(method="min")
    ranked["rank_mae"] = ranked["MAE"].rank(method="min")
    ranked["rank_r2"] = ranked["R2"].rank(method="min", ascending=False)
    ranked["overall_rank"] = ranked[["rank_rmse", "rank_mae", "rank_r2"]].mean(axis=1)
    return ranked.sort_values("overall_rank")


def safe_mape_report(y_true: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    near_zero = int(np.sum(np.abs(y_true) <= 1e-6))
    return {
        "near_zero_excluded_count": near_zero,
        "method": "Exclude |y_true| <= 1e-6 from MAPE denominator",
    }
