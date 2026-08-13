from __future__ import annotations

from sklearn.linear_model import Lasso

from src.config import RANDOM_SEED
from src.models.base_model import SklearnRegressorAdapter


def build_lasso(target_col: str) -> SklearnRegressorAdapter:
    return SklearnRegressorAdapter(name="lasso", estimator=Lasso(alpha=0.001, random_state=RANDOM_SEED, max_iter=5000))
