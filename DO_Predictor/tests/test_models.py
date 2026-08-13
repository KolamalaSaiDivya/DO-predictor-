"""Per-model adapter shape/output verification - every model in the registry
gets fit on a small slice of real data and checked for sane, finite predictions
before it's trusted in the full ablation run."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import PRIMARY_TARGET
from src.models.registry import MODEL_REGISTRY, build_model
from src.pipeline import build_base_artifacts, build_stage_dataframe, prepare_model_data


@pytest.fixture(scope="module")
def stage_df():
    base = build_base_artifacts()
    return build_stage_dataframe(base, PRIMARY_TARGET, "full_features")


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY))
def test_model_fit_predict_shape_and_finiteness(stage_df, model_name):
    splits = prepare_model_data(stage_df, PRIMARY_TARGET, model_name)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    model = build_model(model_name, PRIMARY_TARGET)
    model.fit(X_train, y_train, X_val, y_val)
    assert model.is_fitted

    preds = model.predict(X_test)
    assert len(preds) > 0
    assert len(preds) <= len(X_test)
    assert np.isfinite(preds).all(), f"{model_name} produced non-finite predictions"

    metrics = model.evaluate(X_test, y_test)
    assert np.isfinite(metrics["mae"])
    assert metrics["mae"] >= 0
    assert metrics["n"] > 0

    params = model.get_params()
    assert isinstance(params, dict)
