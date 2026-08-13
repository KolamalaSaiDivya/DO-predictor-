from __future__ import annotations

from sklearn.svm import SVR

from src.models.base_model import SklearnRegressorAdapter


def build_svr(target_col: str) -> SklearnRegressorAdapter:
    # rbf kernel, modest C - full grid search isn't the point here, this is a
    # benchmark comparison not a per-model hyperparameter paper.
    # max_train_samples: RBF SVR training cost blows up past a few thousand
    # rows, capped to the most recent 5000 (still a valid chronological tail).
    return SklearnRegressorAdapter(
        name="svr", estimator=SVR(kernel="rbf", C=10.0, epsilon=0.01), max_train_samples=5000
    )
