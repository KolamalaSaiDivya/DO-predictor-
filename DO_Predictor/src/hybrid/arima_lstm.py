"""ARIMA + LSTM hybrid: LSTM predicts ARIMA residuals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config.research_config import (
    ARIMA_AUTO_ORDER,
    ARIMA_D,
    ARIMA_MAX_D,
    ARIMA_MAX_P,
    ARIMA_MAX_Q,
    ARIMA_P,
    ARIMA_Q,
    LOOKBACK,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_LEARNING_RATE,
    LSTM_UNITS,
    effective_lstm_epochs,
)
from src.logging_utils import setup_experiment_logger
from src.research_evaluation import compute_full_metrics

logger = setup_experiment_logger(__name__)


@dataclass
class HybridArtifacts:
    arima_order: tuple[int, int, int]
    arima_aic: float
    arima_bic: float
    scaler: StandardScaler
    feature_columns: list[str]
    lookback: int
    model: Any
    history: dict[str, list[float]]


def _fit_arima_order(y_train: np.ndarray) -> tuple[tuple[int, int, int], float, float]:
    if not ARIMA_AUTO_ORDER:
        order = (ARIMA_P or 1, ARIMA_D or 1, ARIMA_Q or 1)
        model = SARIMAX(y_train, order=order, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        return order, float(res.aic), float(res.bic)

    best_order = (1, 1, 1)
    best_aic = float("inf")
    best_res = None
    for p in range(0, ARIMA_MAX_P + 1):
        for d in range(0, ARIMA_MAX_D + 1):
            for q in range(0, ARIMA_MAX_Q + 1):
                if p == d == q == 0:
                    continue
                try:
                    model = SARIMAX(y_train, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                        best_res = res
                except Exception:
                    continue
    if best_res is None:
        model = SARIMAX(y_train, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
        best_res = model.fit(disp=False)
        best_order = (1, 1, 1)
    return best_order, float(best_res.aic), float(best_res.bic)


def _arima_in_sample_predictions(y: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    preds = np.full_like(y, np.nan, dtype=float)
    if len(y) < max(10, sum(order) + 5):
        return preds
    model = SARIMAX(y, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fitted = res.get_prediction(start=order[0], end=len(y) - 1)
    preds[order[0] : len(y)] = fitted.predicted_mean
    return preds


def _make_sequence_matrix(features: np.ndarray, targets: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(lookback, len(features)):
        xs.append(features[i - lookback : i])
        ys.append(targets[i])
    if not xs:
        return np.empty((0, lookback, features.shape[1])), np.empty((0,))
    return np.stack(xs), np.asarray(ys)


def train_arima_lstm_hybrid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    lookback: int = LOOKBACK,
) -> tuple[HybridArtifacts, pd.DataFrame, dict[str, float]]:
    from tensorflow import keras

    y_train = train_df[target_col].to_numpy(dtype=float)
    order, aic, bic = _fit_arima_order(y_train)

    combined = pd.concat([train_df, val_df], ignore_index=True)
    y_combined = combined[target_col].to_numpy(dtype=float)
    arima_combined_preds = _arima_in_sample_predictions(y_combined, order)
    combined_residuals = y_combined - arima_combined_preds

    feat_frame = combined[feature_cols].copy()
    feat_frame["arima_fitted"] = arima_combined_preds
    feat_frame["arima_residual_lag1"] = pd.Series(combined_residuals).shift(1)
    feat_frame["target_residual"] = combined_residuals
    feat_frame = feat_frame.dropna().reset_index(drop=True)

    train_len = max(lookback + 1, int(len(train_df) * len(feat_frame) / max(len(combined), 1)))
    train_len = min(train_len, max(lookback + 1, len(feat_frame) - 1))
    train_part = feat_frame.iloc[:train_len]
    val_part = feat_frame.iloc[train_len:]

    model_cols = feature_cols + ["arima_fitted", "arima_residual_lag1"]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_part[model_cols].astype(float))
    val_x = scaler.transform(val_part[model_cols].astype(float)) if len(val_part) else np.empty((0, len(model_cols)))
    train_y = train_part["target_residual"].to_numpy(dtype=float)
    val_y = val_part["target_residual"].to_numpy(dtype=float) if len(val_part) else np.empty((0,))

    X_train, y_seq = _make_sequence_matrix(train_x, train_y, lookback)
    X_val, y_val = (
        _make_sequence_matrix(val_x, val_y, lookback)
        if len(val_x) > lookback
        else (np.empty((0, lookback, len(model_cols))), np.empty((0,)))
    )

    inputs = keras.Input(shape=(lookback, len(model_cols)))
    x = keras.layers.LSTM(LSTM_UNITS)(inputs)
    x = keras.layers.Dropout(LSTM_DROPOUT)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LSTM_LEARNING_RATE), loss="mse", metrics=["mae"])

    callbacks = []
    val_data = None
    if len(X_val) > 0:
        val_data = (X_val, y_val)
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

    history = model.fit(
        X_train,
        y_seq,
        validation_data=val_data,
        epochs=effective_lstm_epochs(),
        batch_size=LSTM_BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )

    artifacts = HybridArtifacts(
        arima_order=order,
        arima_aic=aic,
        arima_bic=bic,
        scaler=scaler,
        feature_columns=model_cols,
        lookback=lookback,
        model=model,
        history=history.history,
    )
    training_summary = {"arima_order": order, "AIC": aic, "BIC": bic, "train_sequences": len(X_train)}
    return artifacts, pd.DataFrame(history.history), training_summary


def predict_arima_lstm_hybrid(
    artifacts: HybridArtifacts,
    test_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_y = pd.concat([train_val_df[target_col], test_df[target_col]], ignore_index=True).to_numpy(dtype=float)
    n_hist = len(train_val_df)

    arima_model = SARIMAX(
        history_y[:n_hist], order=artifacts.arima_order, enforce_stationarity=False, enforce_invertibility=False
    )
    arima_res = arima_model.fit(disp=False)
    forecast = arima_res.get_forecast(steps=len(test_df)).predicted_mean
    arima_forecast = np.asarray(forecast)

    combined = pd.concat([train_val_df, test_df], ignore_index=True)
    arima_insample = _arima_in_sample_predictions(combined[target_col].to_numpy(dtype=float), artifacts.arima_order)
    residuals = combined[target_col].to_numpy(dtype=float) - arima_insample

    feat = combined[feature_cols].copy()
    feat["arima_fitted"] = arima_insample
    feat["arima_residual_lag1"] = pd.Series(residuals).shift(1)
    feat = feat.dropna().reset_index(drop=True)

    scaled = artifacts.scaler.transform(feat[artifacts.feature_columns].astype(float))
    test_start = len(feat) - len(test_df)
    residual_preds = np.full(len(test_df), np.nan, dtype=float)
    for i in range(len(test_df)):
        idx = test_start + i
        if idx < artifacts.lookback:
            continue
        window = scaled[idx - artifacts.lookback : idx]
        residual_preds[i] = float(artifacts.model.predict(window[np.newaxis, ...], verbose=0).reshape(-1)[0])

    hybrid_pred = arima_forecast + np.nan_to_num(residual_preds, nan=0.0)
    return arima_forecast, residual_preds, hybrid_pred


def save_hybrid_outputs(
    output_dir: Path,
    timestamps: pd.Series,
    y_true: np.ndarray,
    arima_pred: np.ndarray,
    lstm_residual: np.ndarray,
    hybrid_pred: np.ndarray,
    artifacts: HybridArtifacts,
    history_df: pd.DataFrame,
    training_time: float,
    prediction_time: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Timestamp": timestamps,
            "actual": y_true,
            "arima_prediction": arima_pred,
            "lstm_residual_prediction": lstm_residual,
            "hybrid_prediction": hybrid_pred,
        }
    ).to_csv(output_dir / "hybrid_predictions.csv", index=False)
    summary = {
        "algorithm": "y_hat_hybrid = y_hat_ARIMA + e_hat_LSTM",
        "arima_order": artifacts.arima_order,
        "AIC": artifacts.arima_aic,
        "BIC": artifacts.arima_bic,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        "metrics": compute_full_metrics(y_true, hybrid_pred),
    }
    (output_dir / "hybrid_model_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    history_df.to_csv(output_dir / "hybrid_training_history.csv", index=False)
    return summary
