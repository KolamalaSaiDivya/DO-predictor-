from __future__ import annotations

from catboost import CatBoostRegressor

from src.config import RANDOM_SEED
from src.models.base_model import SklearnRegressorAdapter


def build_catboost(target_col: str) -> SklearnRegressorAdapter:
    return SklearnRegressorAdapter(
        name="catboost",
        estimator=CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,  # skips the catboost_info/ log dir it dumps by default
        ),
    )
