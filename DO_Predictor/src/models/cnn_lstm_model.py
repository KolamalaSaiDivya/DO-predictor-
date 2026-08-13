from __future__ import annotations

from tensorflow import keras

from src.config import SEQUENCE_LENGTH
from src.models.base_model import SequenceModelAdapter


class CNNLSTMModel(SequenceModelAdapter):
    """1D conv to pick up local shape in the lookback window before the LSTM
    reasons about longer-range dependencies."""

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        return keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu"),
            keras.layers.MaxPooling1D(pool_size=2, padding="same"),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])


def build_cnn_lstm(target_col: str) -> CNNLSTMModel:
    return CNNLSTMModel(name="cnn_lstm", sequence_length=SEQUENCE_LENGTH)
