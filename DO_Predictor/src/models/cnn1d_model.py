from __future__ import annotations

from tensorflow import keras

from src.config import SEQUENCE_LENGTH
from src.models.base_model import SequenceModelAdapter


class CNN1DModel(SequenceModelAdapter):
    """Pure convolutional baseline for the sequence models - no recurrence at
    all, just stacked causal conv layers over the lookback window."""

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        return keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu"),
            keras.layers.Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu"),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])


def build_cnn1d(target_col: str) -> CNN1DModel:
    return CNN1DModel(name="cnn1d", sequence_length=SEQUENCE_LENGTH)
