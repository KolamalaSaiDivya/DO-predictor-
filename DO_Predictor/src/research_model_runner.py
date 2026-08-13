"""Model training/evaluation helpers for research experiments."""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import pandas as pd

from config.research_config import LOOKBACK, TIMESTAMP_COL, effective_lstm_epochs
from src.hybrid.arima_lstm import predict_arima_lstm_hybrid, train_arima_lstm_hybrid
from src.models.registry import SEQUENTIAL_MODELS, build_model
from src.preprocessing import prepare_xy
from src.research_evaluation import compute_full_metrics, timer


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_label: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build X/y ensuring the raw target column is present for persistence models."""
    X, y = prepare_xy(df, feature_cols, target_label)
    if target_col in df.columns and target_col not in X.columns:
        X = X.copy()
        X[target_col] = df[target_col].values
    return X, y


def _build_feature_frame(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    target_label = f"{target_col}_target"
    keep = [TIMESTAMP_COL] if TIMESTAMP_COL in df.columns else []
    keep += [c for c in feature_cols if c in df.columns]
    if target_col in df.columns:
        keep.append(target_col)
    if target_label in df.columns:
        keep.append(target_label)
    return df[keep].dropna().reset_index(drop=True)


def run_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> dict[str, Any]:
    target_label = f"{target_col}_target"
    train = _build_feature_frame(train_df, feature_cols, target_col)
    val = _build_feature_frame(val_df, feature_cols, target_col)
    test = _build_feature_frame(test_df, feature_cols, target_col)

    result: dict[str, Any] = {"model": model_name, "status": "failed"}

    if model_name == "arima_lstm_hybrid":
        with timer() as train_time:
            artifacts, history_df, arima_summary = train_arima_lstm_hybrid(train, val, target_col, feature_cols, LOOKBACK)
        with timer() as pred_time:
            arima_pred, resid_pred, hybrid_pred = predict_arima_lstm_hybrid(
                artifacts, test, pd.concat([train, val], ignore_index=True), target_col, feature_cols
            )
        y_true = test[target_label].to_numpy()
        resid = np.asarray(resid_pred)
        valid = np.isfinite(resid) & (np.arange(len(resid)) >= artifacts.lookback)
        if valid.any():
            y_true = y_true[valid]
            hybrid_pred = hybrid_pred[valid]
            arima_pred = arima_pred[valid]
            resid_pred = resid[valid]
            ts = test[TIMESTAMP_COL].iloc[valid]
        else:
            ts = test[TIMESTAMP_COL]
        metrics = compute_full_metrics(y_true, hybrid_pred)
        result.update(
            {
                "status": "success",
                "metrics": metrics,
                "predictions": hybrid_pred,
                "y_true": y_true,
                "timestamps": ts,
                "training_time": train_time[0],
                "prediction_time": pred_time[0],
                "history_df": history_df,
                "artifacts": artifacts,
                "arima_summary": arima_summary,
                "components": {"arima": arima_pred, "lstm_residual": resid_pred},
            }
        )
        return result

    X_train, y_train = _prepare_xy(train, feature_cols, target_label, target_col)
    X_val, y_val = _prepare_xy(val, feature_cols, target_label, target_col)
    X_test, y_test = _prepare_xy(test, feature_cols, target_label, target_col)

    if model_name in SEQUENTIAL_MODELS and model_name != "persistence":
        from src.pipeline import longest_contiguous_block
        from src.preprocessing import chronological_split as chrono_split

        combined = pd.concat([train, val, test], ignore_index=True)
        combined = longest_contiguous_block(combined)
        train, val, test = chrono_split(combined)
        X_train, y_train = _prepare_xy(train, feature_cols, target_label, target_col)
        X_val, y_val = _prepare_xy(val, feature_cols, target_label, target_col)
        X_test, y_test = _prepare_xy(test, feature_cols, target_label, target_col)

    model = build_model(model_name, target_col)
    if model_name in {"lstm", "gru", "transformer"}:
        model.epochs = effective_lstm_epochs()

    with timer() as train_time:
        model.fit(X_train, y_train, X_val, y_val)
    with timer() as pred_time:
        preds = model.predict(X_test)
        val_preds = model.predict(X_val) if len(X_val) else np.array([])

    y_true = y_test.iloc[-len(preds) :].to_numpy()
    ts = X_test[TIMESTAMP_COL].iloc[-len(preds) :]
    if len(preds) == 0 or len(y_true) == 0:
        result.update(
            {
                "status": "failed",
                "error": "No test samples available after feature alignment.",
            }
        )
        return result
    metrics = compute_full_metrics(y_true, preds)
    val_metrics = (
        compute_full_metrics(y_val.iloc[-len(val_preds) :].to_numpy(), val_preds) if len(val_preds) else {}
    )
    history_df = pd.DataFrame(model.history_) if getattr(model, "history_", None) else pd.DataFrame()

    result.update(
        {
            "status": "success",
            "metrics": metrics,
            "val_metrics": val_metrics,
            "predictions": preds,
            "y_true": y_true,
            "timestamps": ts,
            "training_time": train_time[0],
            "prediction_time": pred_time[0],
            "history_df": history_df,
            "fitted_model": model,
        }
    )
    return result


def safe_run_model(*args, **kwargs) -> dict[str, Any]:
    try:
        return run_model(*args, **kwargs)
    except Exception as exc:
        return {
            "model": kwargs.get("model_name") or (args[0] if args else "unknown"),
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
