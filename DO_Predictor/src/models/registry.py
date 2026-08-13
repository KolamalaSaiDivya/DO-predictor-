"""Single place that maps a model name to a factory function. run_ablation.py
loops over this instead of hardcoding an import list, so adding a model later
means adding one line here, nothing else.
"""

from __future__ import annotations

from typing import Callable

from src.models.arima_model import build_arima
from src.models.base_model import BaseModel
from src.models.bilstm_model import build_bilstm
from src.models.catboost_model import build_catboost
from src.models.cnn1d_model import build_cnn1d
from src.models.cnn_lstm_model import build_cnn_lstm
from src.models.gradient_boosting import build_gradient_boosting
from src.models.gru_model import build_gru
from src.models.knn_model import build_knn
from src.models.lasso_model import build_lasso
from src.models.lightgbm_model import build_lightgbm
from src.models.linear_regression import build_linear_regression
from src.models.lstm_model import build_lstm
from src.models.persistence import build_persistence
from src.models.prophet_model import build_prophet
from src.models.random_forest import build_random_forest
from src.models.ridge_model import build_ridge
from src.models.sarimax_model import build_sarimax
from src.models.svr_model import build_svr
from src.models.transformer_model import build_transformer
from src.models.xgboost_model import build_xgboost

ModelFactory = Callable[[str], BaseModel]

STATISTICAL_MODELS: dict[str, ModelFactory] = {
    "persistence": build_persistence,
    "linear_regression": build_linear_regression,
    "ridge": build_ridge,
    "lasso": build_lasso,
    "arima": build_arima,
    "sarimax": build_sarimax,
    "prophet": build_prophet,
}

CLASSICAL_ML_MODELS: dict[str, ModelFactory] = {
    "svr": build_svr,
    "random_forest": build_random_forest,
    "gradient_boosting": build_gradient_boosting,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "catboost": build_catboost,
    "knn": build_knn,
}

DEEP_LEARNING_MODELS: dict[str, ModelFactory] = {
    "lstm": build_lstm,
    "gru": build_gru,
    "cnn_lstm": build_cnn_lstm,
    "bilstm": build_bilstm,
    "transformer": build_transformer,
    "cnn1d": build_cnn1d,
}

MODEL_REGISTRY: dict[str, ModelFactory] = {
    **STATISTICAL_MODELS,
    **CLASSICAL_ML_MODELS,
    **DEEP_LEARNING_MODELS,
}

# which models need a genuinely contiguous (no time-gap) series rather than an
# arbitrary row-wise feature matrix - pipeline.py uses this to decide which
# train/test slice to hand each model. Sequence DL models build sliding
# windows, so a gap in the underlying rows would silently stitch together
# unrelated time periods into one training example.
SEQUENTIAL_MODELS = {"arima", "sarimax", "prophet", *DEEP_LEARNING_MODELS}


def build_model(name: str, target_col: str) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known models: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](target_col)
