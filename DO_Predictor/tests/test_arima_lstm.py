"""Tests for ARIMA-LSTM hybrid."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.research_config import apply_mode
from src.hybrid.arima_lstm import predict_arima_lstm_hybrid, train_arima_lstm_hybrid


@pytest.fixture(scope="module", autouse=True)
def fast_mode():
    apply_mode(fast=True)
    yield
    apply_mode(full=True)


def _synthetic_frames(n: int = 200):
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="10min")
    df = pd.DataFrame(
        {
            "Timestamp": ts,
            "pH": 7.0 + rng.normal(scale=0.1, size=n).cumsum() * 0.01,
            "Temperature": 20 + rng.normal(scale=0.5, size=n),
        }
    )
    train = df.iloc[:140]
    val = df.iloc[140:170]
    test = df.iloc[170:]
    features = ["Temperature"]
    return train, val, test, features


def test_arima_lstm_train_predict():
    train, val, test, features = _synthetic_frames()
    artifacts, history, summary = train_arima_lstm_hybrid(train, val, "pH", features, lookback=5)
    arima_p, resid_p, hybrid_p = predict_arima_lstm_hybrid(
        artifacts, test, pd.concat([train, val], ignore_index=True), "pH", features
    )
    assert len(hybrid_p) == len(test)
    assert len(history) > 0
    assert "AIC" in summary
