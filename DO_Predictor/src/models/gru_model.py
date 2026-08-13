from __future__ import annotations

from tensorflow import keras

from src.config import SEQUENCE_LENGTH
from src.models.base_model import SequenceModelAdapter


class GRUModel(SequenceModelAdapter):
    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        return keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.GRU(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.GRU(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])


def build_gru(target_col: str) -> GRUModel:
    return GRUModel(name="gru", sequence_length=SEQUENCE_LENGTH)
