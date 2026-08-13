"""RFE, CFS, and combined RFE+CFS feature selection (train-only fitting)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error

from config.research_config import CFS_MAX_FEATURES, RFE_FEATURE_COUNTS, RFE_STEP
from src.evaluation import compute_metrics
from src.logging_utils import setup_experiment_logger

logger = setup_experiment_logger(__name__)


@dataclass
class FeatureSelectionResult:
    method: str
    selected_features: list[str]
    rankings: pd.DataFrame
    validation_scores: pd.DataFrame | None = None
    metadata: dict[str, Any] | None = None


def _estimator() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


def run_rfe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_counts: list[int] | None = None,
    step: int = RFE_STEP,
) -> FeatureSelectionResult:
    feature_counts = feature_counts or RFE_FEATURE_COUNTS
    feature_counts = sorted({c for c in feature_counts if c <= X_train.shape[1] and c >= 1})
    if not feature_counts:
        feature_counts = [min(10, X_train.shape[1])]

    full_ranking = pd.DataFrame({"feature_name": X_train.columns})
    best_count = feature_counts[0]
    best_rmse = float("inf")
    val_scores: list[dict[str, float | int]] = []

    for n_features in feature_counts:
        selector = RFE(estimator=_estimator(), n_features_to_select=n_features, step=step)
        selector.fit(X_train, y_train)
        selected = list(np.array(X_train.columns)[selector.support_])
        model = _estimator()
        model.fit(X_train[selected], y_train)
        preds = model.predict(X_val[selected])
        metrics = compute_metrics(y_val.to_numpy(), preds)
        val_scores.append({"feature_count": n_features, "rmse": metrics["rmse"], "mae": metrics["mae"]})
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_count = n_features
            full_ranking = pd.DataFrame(
                {
                    "feature_name": X_train.columns,
                    "ranking": selector.ranking_,
                    "selected": selector.support_,
                }
            ).sort_values("ranking")

    final_selector = RFE(estimator=_estimator(), n_features_to_select=best_count, step=step)
    final_selector.fit(X_train, y_train)
    selected_features = list(np.array(X_train.columns)[final_selector.support_])
    rankings = pd.DataFrame(
        {
            "feature_name": X_train.columns,
            "ranking": final_selector.ranking_,
            "selected": final_selector.support_,
        }
    ).sort_values("ranking")

    return FeatureSelectionResult(
        method="RFE",
        selected_features=selected_features,
        rankings=rankings,
        validation_scores=pd.DataFrame(val_scores),
        metadata={"best_feature_count": best_count, "best_validation_rmse": best_rmse},
    )


def cfs_merit(corr_matrix: np.ndarray, feature_indices: list[int], target_idx: int) -> float:
    """Correlation-based Feature Selection merit function using absolute correlations."""
    k = len(feature_indices)
    if k == 0:
        return 0.0
    r_cf = float(np.mean([abs(corr_matrix[i, target_idx]) for i in feature_indices]))
    if k == 1:
        return r_cf
    ff_vals = [
        abs(corr_matrix[feature_indices[i], feature_indices[j]])
        for i in range(k)
        for j in range(i + 1, k)
    ]
    r_ff = float(np.mean(ff_vals)) if ff_vals else 0.0
    return float((k * r_cf) / np.sqrt(k + k * (k - 1) * r_ff))


def run_cfs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_features: int | None = None,
) -> FeatureSelectionResult:
    max_features = max_features or CFS_MAX_FEATURES or min(30, X_train.shape[1])
    max_features = min(max_features, X_train.shape[1])

    corr = X_train.corrwith(y_train).abs().fillna(0.0)
    combined = pd.concat([X_train, y_train.rename("_target")], axis=1)
    corr_matrix = combined.corr().abs().fillna(0.0).to_numpy()
    feature_names = list(X_train.columns)
    target_idx = len(feature_names)

    selected: list[int] = []
    remaining = list(range(len(feature_names)))
    scores: list[dict[str, float | str | bool]] = []

    while len(selected) < max_features and remaining:
        best_idx = None
        best_merit = -1.0
        for idx in remaining:
            trial = selected + [idx]
            merit = cfs_merit(corr_matrix, trial, target_idx)
            if merit > best_merit:
                best_merit = merit
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        scores.append(
            {
                "feature_name": feature_names[best_idx],
                "merit": best_merit,
                "selected": True,
            }
        )

    selected_features = [feature_names[i] for i in selected]
    score_df = pd.DataFrame(scores)
    rankings = pd.DataFrame(
        {
            "feature_name": feature_names,
            "target_correlation": [float(corr.get(c, 0.0)) for c in feature_names],
            "selected": [c in selected_features for c in feature_names],
        }
    ).sort_values("target_correlation", ascending=False)

    return FeatureSelectionResult(
        method="CFS",
        selected_features=selected_features,
        rankings=rankings,
        metadata={"max_features": max_features, "cfs_scores": score_df.to_dict(orient="records")},
    )


def run_rfe_cfs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_counts: list[int] | None = None,
    cfs_max_features: int | None = None,
    rfe_step: int = RFE_STEP,
) -> tuple[FeatureSelectionResult, FeatureSelectionResult, FeatureSelectionResult]:
    rfe_result = run_rfe(X_train, y_train, X_val, y_val, feature_counts=feature_counts, step=rfe_step)
    rfe_cols = rfe_result.selected_features
    max_cfs = cfs_max_features or CFS_MAX_FEATURES or len(rfe_cols)
    cfs_result = run_cfs(X_train[rfe_cols], y_train, max_features=min(len(rfe_cols), max_cfs))
    final_features = cfs_result.selected_features
    combined = FeatureSelectionResult(
        method="RFE+CFS",
        selected_features=final_features,
        rankings=cfs_result.rankings,
        metadata={
            "initial_feature_count": X_train.shape[1],
            "rfe_feature_count": len(rfe_cols),
            "cfs_feature_count": len(final_features),
            "final_feature_count": len(final_features),
            "rfe_selected": rfe_cols,
            "removed_features": [c for c in X_train.columns if c not in final_features],
        },
    )
    return rfe_result, cfs_result, combined


def save_feature_selection_outputs(
    output_dir: Path,
    rfe: FeatureSelectionResult,
    cfs: FeatureSelectionResult,
    combined: FeatureSelectionResult,
    periods: dict[str, str | None],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rfe.rankings.to_csv(output_dir / "rfe_rankings.csv", index=False)
    pd.DataFrame({"feature_name": rfe.selected_features}).to_csv(output_dir / "rfe_selected_features.csv", index=False)
    if rfe.validation_scores is not None:
        rfe.validation_scores.to_csv(output_dir / "rfe_validation_scores.csv", index=False)

    cfs.rankings.to_csv(output_dir / "cfs_scores.csv", index=False)
    pd.DataFrame({"feature_name": cfs.selected_features}).to_csv(output_dir / "cfs_selected_features.csv", index=False)

    pd.DataFrame({"feature_name": combined.selected_features}).to_csv(output_dir / "final_rfe_cfs_features.csv", index=False)
    report = {
        "initial_feature_count": combined.metadata.get("initial_feature_count") if combined.metadata else None,
        "rfe_feature_count": combined.metadata.get("rfe_feature_count") if combined.metadata else None,
        "cfs_feature_count": combined.metadata.get("cfs_feature_count") if combined.metadata else None,
        "final_feature_count": len(combined.selected_features),
        "selected_features": combined.selected_features,
        "removed_features": combined.metadata.get("removed_features") if combined.metadata else [],
        "selection_method": "RFE then CFS",
        **periods,
    }
    (output_dir / "final_feature_selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
