from __future__ import annotations

from tensorflow import keras

from src.config import SEQUENCE_LENGTH
from src.models.base_model import SequenceModelAdapter


class BiLSTMModel(SequenceModelAdapter):
    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        return keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Bidirectional(keras.layers.LSTM(48, return_sequences=True)),
            keras.layers.Dropout(0.2),
            keras.layers.Bidirectional(keras.layers.LSTM(24)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])


def build_bilstm(target_col: str) -> BiLSTMModel:
    return BiLSTMModel(name="bilstm", sequence_length=SEQUENCE_LENGTH)
