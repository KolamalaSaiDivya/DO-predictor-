from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from src.config import RANDOM_SEED
from src.models.base_model import SklearnRegressorAdapter


def build_random_forest(target_col: str) -> SklearnRegressorAdapter:
    return SklearnRegressorAdapter(
        name="random_forest",
        estimator=RandomForestRegressor(
            n_estimators=200, max_depth=14, n_jobs=-1, random_state=RANDOM_SEED
        ),
    )
