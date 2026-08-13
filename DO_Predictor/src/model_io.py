"""Save/load trained models to trained_models/. Keras models need their own
save format; everything else (sklearn adapters, statsmodels, Prophet) is a
plain joblib pickle."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.config import TRAINED_MODELS_DIR, get_logger
from src.models.base_model import BaseModel, SequenceModelAdapter
from src.models.registry import build_model

logger = get_logger(__name__)


def _base_path(target_col: str, model_name: str, out_dir: Path) -> Path:
    safe_target = target_col.replace(" ", "_").replace("%", "pct").replace("(", "").replace(")", "")
    return out_dir / f"{safe_target}__{model_name}"


def save_model(model: BaseModel, target_col: str, out_dir: Path = TRAINED_MODELS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _base_path(target_col, model.name, out_dir)

    if isinstance(model, SequenceModelAdapter):
        keras_path = base.with_suffix(".keras")
        model.model_.save(keras_path)
        joblib.dump(
            {
                "scaler": model.scaler_,
                "feature_columns": model.feature_columns_,
                "sequence_length": model.sequence_length,
                "params": model.params,
            },
            base.with_suffix(".meta.joblib"),
        )
        logger.info("Saved %s to %s (+ meta)", model.name, keras_path)
        return keras_path

    joblib_path = base.with_suffix(".joblib")
    joblib.dump(model, joblib_path)
    logger.info("Saved %s to %s", model.name, joblib_path)
    return joblib_path


def load_model(model_name: str, target_col: str, out_dir: Path = TRAINED_MODELS_DIR) -> BaseModel:
    base = _base_path(target_col, model_name, out_dir)
    keras_path = base.with_suffix(".keras")

    if keras_path.exists():
        from tensorflow import keras

        model = build_model(model_name, target_col)
        model.model_ = keras.models.load_model(keras_path)
        meta = joblib.load(base.with_suffix(".meta.joblib"))
        model.scaler_ = meta["scaler"]
        model.feature_columns_ = meta["feature_columns"]
        model.is_fitted = True
        return model

    joblib_path = base.with_suffix(".joblib")
    if not joblib_path.exists():
        raise FileNotFoundError(f"No saved model for {model_name}/{target_col} in {out_dir}")
    return joblib.load(joblib_path)


def model_exists(model_name: str, target_col: str, out_dir: Path = TRAINED_MODELS_DIR) -> bool:
    base = _base_path(target_col, model_name, out_dir)
    return base.with_suffix(".joblib").exists() or base.with_suffix(".keras").exists()
