"""Uniform fit/predict/evaluate/get_params interface, plus the adapters that
let statsmodels/Prophet/Keras models plug into the same registry as sklearn
regressors even though their input shapes are all different.

Convention used everywhere below: predictions always align with the *trailing*
rows of whatever X was passed in. Sklearn-style models predict one row per
input row, so that's trivial. Sequence models can only start predicting once
they have `sequence_length` rows of history, so they return fewer predictions
than len(X) - still the trailing slice, so `evaluate()` can stay generic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_logger
from src.evaluation import compute_metrics

logger = get_logger(__name__)


class BaseModel(ABC):
    """Every model in the registry subclasses this, directly or via an adapter."""

    def __init__(self, name: str, **params: Any) -> None:
        self.name = name
        self.params = params
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        preds = self.predict(X)
        y_true = y.iloc[-len(preds):] if len(preds) < len(y) else y
        return compute_metrics(y_true.to_numpy(), preds)

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class SklearnRegressorAdapter(BaseModel):
    """Wraps anything exposing sklearn's .fit(X, y)/.predict(X) - covers
    LinearRegression, Ridge, Lasso, SVR, RandomForest, GradientBoosting,
    XGBoost, LightGBM, CatBoost, KNN. They all speak the same API so one
    adapter does for all of them.
    """

    def __init__(
        self,
        name: str,
        estimator: Any,
        exclude_columns: list[str] | None = None,
        max_train_samples: int | None = None,
        **params: Any,
    ) -> None:
        super().__init__(name, **params)
        self.estimator = estimator
        self.exclude_columns = exclude_columns or ["Timestamp"]
        self.feature_columns_: list[str] | None = None
        # kernel SVR is O(n^2)-O(n^3) - on ~15k rows that's minutes-to-hours.
        # Capping to the most recent max_train_samples rows (still the tail of
        # a chronological split, so no leakage) keeps it tractable without
        # dropping the model from the registry.
        self.max_train_samples = max_train_samples

    def _feature_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self.feature_columns_ or [c for c in X.columns if c not in self.exclude_columns]
        return X[cols].astype(float)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "SklearnRegressorAdapter":
        self.feature_columns_ = [c for c in X_train.columns if c not in self.exclude_columns]
        if self.max_train_samples is not None and len(X_train) > self.max_train_samples:
            X_train = X_train.iloc[-self.max_train_samples:]
            y_train = y_train.iloc[-self.max_train_samples:]
        self.estimator.fit(self._feature_frame(X_train).to_numpy(), y_train.to_numpy())
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(self._feature_frame(X).to_numpy()))

    def get_params(self) -> dict[str, Any]:
        try:
            return self.estimator.get_params()
        except AttributeError:
            return dict(self.params)

    def feature_importances(self) -> dict[str, float] | None:
        """Only meaningful for tree ensembles - used by notebook 3."""
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None or self.feature_columns_ is None:
            return None
        return dict(zip(self.feature_columns_, importances.tolist()))


class PersistenceModel(BaseModel):
    """DO(t+1) = DO(t). No fitting, just carries the current reading forward -
    the floor every other model needs to beat."""

    def __init__(self, name: str, target_col: str, **params: Any) -> None:
        super().__init__(name, **params)
        self.target_col = target_col

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "PersistenceModel":
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.target_col].to_numpy()


class StatsForecastAdapter(BaseModel):
    """Base for ARIMA/SARIMAX. Fits once on the train series, then forecasts
    the whole eval horizon in one shot via get_forecast(steps=...). Only valid
    when X passed to predict() picks up chronologically right where training
    left off - that's how pipeline.py calls it, never shuffled.
    """

    def __init__(
        self,
        name: str,
        order: tuple[int, int, int] = (2, 1, 2),
        seasonal_order: tuple[int, int, int, int] | None = None,
        exog_columns: list[str] | None = None,
        **params: Any,
    ) -> None:
        super().__init__(name, order=order, seasonal_order=seasonal_order, exog_columns=exog_columns, **params)
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_columns = exog_columns
        self.results_ = None

    def _exog(self, X: pd.DataFrame) -> np.ndarray | None:
        if not self.exog_columns:
            return None
        return X[self.exog_columns].astype(float).to_numpy()

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "StatsForecastAdapter":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            y_train.to_numpy(),
            exog=self._exog(X_train),
            order=self.order,
            seasonal_order=self.seasonal_order or (0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.results_ = model.fit(disp=False)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        forecast = self.results_.get_forecast(steps=len(X), exog=self._exog(X))
        return np.asarray(forecast.predicted_mean)


class ProphetAdapter(BaseModel):
    """Plain univariate Prophet - trend + yearly/weekly/daily seasonality,
    no extra regressors, matching how Prophet is conventionally used (and
    how the cited public baseline was almost certainly set up)."""

    def __init__(self, name: str, timestamp_col: str = "Timestamp", **prophet_kwargs: Any) -> None:
        super().__init__(name, **prophet_kwargs)
        self.timestamp_col = timestamp_col
        self.prophet_kwargs = prophet_kwargs
        self.model_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "ProphetAdapter":
        from prophet import Prophet

        train_df = pd.DataFrame({"ds": X_train[self.timestamp_col].to_numpy(), "y": y_train.to_numpy()})
        self.model_ = Prophet(**self.prophet_kwargs)
        self.model_.fit(train_df)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        future = pd.DataFrame({"ds": X[self.timestamp_col].to_numpy()})
        forecast = self.model_.predict(future)
        return forecast["yhat"].to_numpy()


class SequenceModelAdapter(BaseModel):
    """Base for the Keras models (LSTM/GRU/CNN-LSTM/BiLSTM/Transformer/1D-CNN).

    Handles scaling + windowing + training loop; subclasses just implement
    `_build_model(input_shape)`. predict() needs `sequence_length` rows of
    history before the first row it can score, so it returns len(X) -
    sequence_length predictions (the trailing slice, per the module convention).
    """

    def __init__(
        self,
        name: str,
        sequence_length: int,
        epochs: int = 40,
        batch_size: int = 64,
        patience: int = 6,
        exclude_columns: list[str] | None = None,
        **params: Any,
    ) -> None:
        super().__init__(name, sequence_length=sequence_length, epochs=epochs, batch_size=batch_size, **params)
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.exclude_columns = exclude_columns or ["Timestamp"]
        self.feature_columns_: list[str] | None = None
        self.scaler_ = None
        self.model_ = None
        self.history_: dict[str, list[float]] | None = None

    def _build_model(self, input_shape: tuple[int, int]):
        raise NotImplementedError("Subclasses (lstm_model.py etc.) must implement _build_model.")

    def _feature_frame(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_columns_ or [c for c in X.columns if c not in self.exclude_columns]
        return X[cols].astype(float).to_numpy()

    def _make_windows(self, features: np.ndarray, y: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        n = len(features)
        seq_len = self.sequence_length
        if n <= seq_len:
            empty_x = np.empty((0, seq_len, features.shape[1]))
            return (empty_x, np.empty((0,)) if y is not None else None)

        windows = np.stack([features[i - seq_len:i] for i in range(seq_len, n)])
        targets = y[seq_len:] if y is not None else None
        return windows, targets

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "SequenceModelAdapter":
        from sklearn.preprocessing import StandardScaler
        from tensorflow import keras

        self.feature_columns_ = [c for c in X_train.columns if c not in self.exclude_columns]
        self.scaler_ = StandardScaler()
        train_features = self.scaler_.fit_transform(self._feature_frame(X_train))
        X_seq, y_seq = self._make_windows(train_features, y_train.to_numpy())

        val_data = None
        if X_val is not None and y_val is not None:
            val_features = self.scaler_.transform(self._feature_frame(X_val))
            X_val_seq, y_val_seq = self._make_windows(val_features, y_val.to_numpy())
            if len(X_val_seq) > 0:
                val_data = (X_val_seq, y_val_seq)

        self.model_ = self._build_model(input_shape=(self.sequence_length, train_features.shape[1]))
        self.model_.compile(optimizer="adam", loss="mse", metrics=["mae"])

        callbacks = []
        if val_data is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
            )

        history = self.model_.fit(
            X_seq,
            y_seq,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        self.history_ = history.history
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        features = self.scaler_.transform(self._feature_frame(X))
        X_seq, _ = self._make_windows(features, None)
        if len(X_seq) == 0:
            return np.array([])
        preds = self.model_.predict(X_seq, verbose=0)
        return np.asarray(preds).reshape(-1)
