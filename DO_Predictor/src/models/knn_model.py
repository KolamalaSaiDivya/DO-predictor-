from __future__ import annotations

from sklearn.neighbors import KNeighborsRegressor

from src.models.base_model import SklearnRegressorAdapter


def build_knn(target_col: str) -> SklearnRegressorAdapter:
    return SklearnRegressorAdapter(name="knn", estimator=KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1))
