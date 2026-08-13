from __future__ import annotations

from sklearn.linear_model import LinearRegression

from src.models.base_model import SklearnRegressorAdapter


def build_linear_regression(target_col: str) -> SklearnRegressorAdapter:
    return SklearnRegressorAdapter(name="linear_regression", estimator=LinearRegression())
