"""Metrics, residual analysis, and per-stage error logging shared by every model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import get_logger

logger = get_logger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sklearn's MAPE blows up near y_true=0; DO never hits zero in this
    dataset so a plain implementation is fine, but guard against it anyway."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.abs(y_true) > 1e-6
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "r2": float("nan"), "n": 0}
    y_true, y_pred = y_true[mask], y_pred[mask]

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if mask.sum() > 1 else float("nan"),
        "n": int(mask.sum()),
    }


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)


def compare_to_baselines(mae: float, baseline_mae: dict[str, float]) -> dict[str, Any]:
    """% improvement of a model's MAE over each named baseline (positive = better)."""
    out = {}
    for name, base_mae in baseline_mae.items():
        out[f"vs_{name}_mae"] = base_mae
        out[f"vs_{name}_pct_improvement"] = round((base_mae - mae) / base_mae * 100, 2) if base_mae else float("nan")
    return out


class StageMetricLogger:
    """Collects metrics keyed by (pipeline_stage, target, model_name) so
    run_ablation.py can dump one flat structure at the end."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log(self, stage: str, target: str, model_name: str, split: str, metrics: dict[str, float], **extra: Any) -> None:
        record = {"stage": stage, "target": target, "model": model_name, "split": split, **metrics, **extra}
        self.records.append(record)
        logger.info(
            "[%s][%s][%s][%s] MAE=%.4f RMSE=%.4f MAPE=%.2f%% R2=%.4f (n=%d)",
            stage, target, model_name, split,
            metrics.get("mae", float("nan")),
            metrics.get("rmse", float("nan")),
            metrics.get("mape", float("nan")),
            metrics.get("r2", float("nan")),
            metrics.get("n", 0),
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def to_dict_list(self) -> list[dict[str, Any]]:
        return self.records
