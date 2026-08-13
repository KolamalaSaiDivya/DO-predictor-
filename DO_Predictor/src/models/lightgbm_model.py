from __future__ import annotations

from lightgbm import LGBMRegressor

from src.config import RANDOM_SEED
from src.models.base_model import SklearnRegressorAdapter


def build_lightgbm(target_col: str) -> SklearnRegressorAdapter:
    return SklearnRegressorAdapter(
        name="lightgbm",
        estimator=LGBMRegressor(
            n_estimators=300,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbose=-1,
        ),
    )
