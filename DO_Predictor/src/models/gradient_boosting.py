from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor

from src.config import RANDOM_SEED
from src.models.base_model import SklearnRegressorAdapter


def build_gradient_boosting(target_col: str) -> SklearnRegressorAdapter:
    return SklearnRegressorAdapter(
        name="gradient_boosting",
        estimator=GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED
        ),
    )
