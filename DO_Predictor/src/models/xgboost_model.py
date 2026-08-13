from __future__ import annotations

from xgboost import XGBRegressor

from src.config import RANDOM_SEED
from src.models.base_model import SklearnRegressorAdapter


def build_xgboost(target_col: str) -> SklearnRegressorAdapter:
    return SklearnRegressorAdapter(
        name="xgboost",
        estimator=XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
    )
