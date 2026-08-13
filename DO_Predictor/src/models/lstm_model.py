from __future__ import annotations

from tensorflow import keras

from src.config import SEQUENCE_LENGTH
from src.models.base_model import SequenceModelAdapter


class LSTMModel(SequenceModelAdapter):
    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        return keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])


def build_lstm(target_col: str) -> LSTMModel:
    return LSTMModel(name="lstm", sequence_length=SEQUENCE_LENGTH)
