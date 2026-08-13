from __future__ import annotations

from sklearn.linear_model import Ridge

from src.config import RANDOM_SEED
from src.models.base_model import SklearnRegressorAdapter


def build_ridge(target_col: str) -> SklearnRegressorAdapter:
    return SklearnRegressorAdapter(name="ridge", estimator=Ridge(alpha=1.0, random_state=RANDOM_SEED))
