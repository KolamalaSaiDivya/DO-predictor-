from __future__ import annotations

import numpy as np
from tensorflow import keras

from src.config import SEQUENCE_LENGTH
from src.models.base_model import SequenceModelAdapter


def _sinusoidal_positions(seq_len: int, d_model: int) -> np.ndarray:
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


class _EncoderBlock(keras.layers.Layer):
    """Standard pre-norm transformer encoder block - attention doesn't come
    with any notion of order on its own, that's what the positional encoding
    added before this block is for."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(d_model),
        ])
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ff_out = self.ff(x)
        return self.norm2(x + self.drop2(ff_out, training=training))


class TransformerModel(SequenceModelAdapter):
    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        seq_len, n_features = input_shape
        d_model = 32

        inputs = keras.layers.Input(shape=input_shape)
        x = keras.layers.Dense(d_model)(inputs)
        pos_encoding = keras.ops.convert_to_tensor(_sinusoidal_positions(seq_len, d_model), dtype="float32")
        x = x + pos_encoding
        x = _EncoderBlock(d_model=d_model, num_heads=4, ff_dim=64)(x)
        x = _EncoderBlock(d_model=d_model, num_heads=4, ff_dim=64)(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(16, activation="relu")(x)
        outputs = keras.layers.Dense(1)(x)
        return keras.Model(inputs, outputs)


def build_transformer(target_col: str) -> TransformerModel:
    return TransformerModel(name="transformer", sequence_length=SEQUENCE_LENGTH)
