"""Chapter 5 experiment: DO forecasting with RFE+CFS feature selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.research_config import (
    CHAPTER5_DATA_PATH,
    CHAPTER5_RESULTS,
    CHAPTER5_TARGET,
    EXCLUDE_FROM_FEATURES,
    FAST_MODE,
    LOW_DO_THRESHOLD,
    MIN_FEATURE_COVERAGE,
    TIMESTAMP_COLUMN_CHAPTER5,
    WARNING_DO_THRESHOLD,
    effective_cfs_max_features,
    effective_chapter5_feature_sets,
    effective_chapter5_models,
    effective_lag_steps,
    effective_max_rows,
    effective_rfe_feature_counts,
    effective_rfe_step,
    effective_rolling_windows,
)
from src.data_quality import preprocess_dataset
from src.feature_selection import run_cfs, run_rfe, run_rfe_cfs, save_feature_selection_outputs
from src.forecasting_utils import compute_horizon_steps
from src.logging_utils import setup_experiment_logger
from src.plotting import (
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_feature_count_vs_error,
    plot_feature_merit,
    plot_metric_comparison,
    plot_time_series,
)
from src.preprocessing import chronological_split, drop_incomplete_feature_rows, drop_na_features, filter_features_by_coverage, split_period_strings, subsample_for_fast_mode
from src.reproducibility import set_global_seeds
from src.research_data_loader import get_numeric_sensor_columns, load_research_dataset, save_load_report
from src.research_evaluation import rank_models
from src.research_feature_engineering import add_forecast_target, create_temporal_features, list_feature_columns
from src.research_model_runner import safe_run_model
from src.results_export import save_dataframe, write_excel_sheets, write_json

logger = setup_experiment_logger(__name__, CHAPTER5_RESULTS / "logs" / "chapter5.log")


def _generate_alerts(pred_df: pd.DataFrame) -> pd.DataFrame:
    if LOW_DO_THRESHOLD is None:
        print("LOW_DO_THRESHOLD must be configured from the study's adopted aquaculture criterion.")
        return pd.DataFrame(
            columns=["timestamp", "actual_DO", "predicted_DO", "status", "threshold", "lead_time_minutes"]
        )

    rows = []
    for _, row in pred_df.iterrows():
        predicted = row["predicted_DO"]
        if predicted < LOW_DO_THRESHOLD:
            status = "Low DO Alert"
        elif WARNING_DO_THRESHOLD is not None and predicted < WARNING_DO_THRESHOLD:
            status = "Warning"
        else:
            status = "Normal"
        rows.append(
            {
                "timestamp": row["timestamp_target"],
                "actual_DO": row["actual_DO"],
                "predicted_DO": predicted,
                "status": status,
                "threshold": LOW_DO_THRESHOLD,
                "lead_time_minutes": row.get("forecast_horizon_minutes"),
            }
        )
    return pd.DataFrame(rows)


def run_chapter5() -> dict[str, Any]:
    set_global_seeds()
    summary: dict[str, Any] = {"status": "failed", "chapter": "chapter5_do"}

    if not CHAPTER5_DATA_PATH.exists():
        msg = f"Chapter 5 dataset not found at {CHAPTER5_DATA_PATH}."
        logger.error(msg)
        summary["error"] = msg
        return summary

    raw, load_report = load_research_dataset(CHAPTER5_DATA_PATH, timestamp_override=TIMESTAMP_COLUMN_CHAPTER5)
    save_load_report(load_report, CHAPTER5_RESULTS / "data" / "load_report.json")
    raw = subsample_for_fast_mode(raw, effective_max_rows())

    target_col = CHAPTER5_TARGET
    if target_col not in raw.columns:
        summary["error"] = f"Target column '{target_col}' not found. Expected canonical name 'DO'."
        logger.error(summary["error"])
        return summary

    cleaned, prep_summary, _ = preprocess_dataset(raw, target_col)
    prep_summary.to_csv(CHAPTER5_RESULTS / "data" / "preprocessing_summary.csv", index=False)
    if not FAST_MODE:
        cleaned.to_csv(CHAPTER5_RESULTS / "data" / "cleaned_dataset.csv", index=False)

    horizon = compute_horizon_steps(cleaned)
    write_json(CHAPTER5_RESULTS / "data" / "horizon_report.json", horizon.to_dict())

    sensor_cols = get_numeric_sensor_columns(cleaned, target_col)
    if "pH" not in sensor_cols and "pH" in cleaned.columns:
        sensor_cols.append("pH")

    featured, feature_dictionary = create_temporal_features(
        cleaned,
        sensor_cols,
        lags=effective_lag_steps(),
        rolling_windows=effective_rolling_windows(),
    )
    featured = add_forecast_target(featured, target_col, horizon_steps=horizon.horizon_steps)
    featured = drop_na_features(featured, target_col)

    all_features = list_feature_columns(featured, EXCLUDE_FROM_FEATURES + [target_col, f"{target_col}_target"])
    all_features = filter_features_by_coverage(featured, all_features, MIN_FEATURE_COVERAGE)
    featured = drop_incomplete_feature_rows(featured, all_features, target_col)
    if featured.empty:
        summary["error"] = "No rows remain after removing incomplete feature rows."
        logger.error(summary["error"])
        return summary
    if not FAST_MODE:
        featured.to_csv(CHAPTER5_RESULTS / "data" / "do_all_candidate_features.csv", index=False)
    feature_dictionary.to_csv(CHAPTER5_RESULTS / "data" / "do_feature_dictionary.csv", index=False)

    train, val, test = chronological_split(featured)
    periods = split_period_strings(train, val, test)

    X_train = train[all_features].astype(float)
    y_train = train[f"{target_col}_target"].astype(float)
    X_val = val[all_features].astype(float)
    y_val = val[f"{target_col}_target"].astype(float)

    rfe_counts = effective_rfe_feature_counts()
    rfe_result, cfs_result, combined_result = run_rfe_cfs(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_counts=rfe_counts,
        cfs_max_features=effective_cfs_max_features(),
        rfe_step=effective_rfe_step(len(all_features)),
    )
    if FAST_MODE:
        logger.info("FAST_MODE active — results are for smoke testing only.")
    save_feature_selection_outputs(
        CHAPTER5_RESULTS / "feature_selection",
        rfe_result,
        cfs_result,
        combined_result,
        periods,
    )

    if rfe_result.validation_scores is not None:
        plot_feature_count_vs_error(
            rfe_result.validation_scores, CHAPTER5_RESULTS / "figures" / "feature_count_vs_validation_error"
        )
    if cfs_result.metadata and cfs_result.metadata.get("cfs_scores"):
        plot_feature_merit(pd.DataFrame(cfs_result.metadata["cfs_scores"]), CHAPTER5_RESULTS / "figures" / "cfs_feature_merit")

    plot_correlation_heatmap(
        train[all_features[: min(30, len(all_features))]].corr(),
        "Candidate Feature Correlation",
        CHAPTER5_RESULTS / "figures" / "correlation_heatmap",
    )
    plot_time_series(cleaned, "Timestamp", target_col, "DO Time Series", CHAPTER5_RESULTS / "figures" / "do_timeseries")
    if "pH" in cleaned.columns:
        plot_time_series(cleaned, "Timestamp", "pH", "pH Time Series", CHAPTER5_RESULTS / "figures" / "ph_timeseries")

    feature_sets = {
        "all_features": all_features,
        "rfe": rfe_result.selected_features,
        "cfs": cfs_result.selected_features,
        "rfe_cfs": combined_result.selected_features,
    }
    active_sets = {k: v for k, v in feature_sets.items() if k in effective_chapter5_feature_sets()}

    comparison_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    final_predictions: pd.DataFrame | None = None
    val_selection_rows: list[dict[str, Any]] = []

    for feature_set_name, features in active_sets.items():
        for model_name in effective_chapter5_models():
            result = safe_run_model(model_name, train, val, test, target_col, features)
            if result["status"] != "success":
                failures.append({"Feature_Set": feature_set_name, "Model": model_name, "error": result.get("error", "")})
                continue
            metrics = result["metrics"]
            comparison_rows.append(
                {
                    "Feature_Set": feature_set_name,
                    "Number_of_Features": len(features),
                    "Model": model_name,
                    "MAE": metrics["MAE"],
                    "MSE": metrics["MSE"],
                    "RMSE": metrics["RMSE"],
                    "MAPE": metrics["MAPE"],
                    "R2": metrics["R2"],
                    "Training_Time": result["training_time"],
                    "Prediction_Time": result["prediction_time"],
                }
            )
            if feature_set_name == "rfe_cfs" and result.get("val_metrics"):
                val_selection_rows.append({"Model": model_name, **result["val_metrics"]})

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(CHAPTER5_RESULTS / "metrics" / "chapter5_feature_selection_model_comparison.csv", index=False)

    best_model = None
    best_features = combined_result.selected_features
    final_metrics: dict[str, float] = {}
    if val_selection_rows:
        val_df = pd.DataFrame(val_selection_rows).sort_values("RMSE")
        best_model = val_df.iloc[0]["Model"]
        final_result = safe_run_model(best_model, train, val, test, target_col, best_features)
        if final_result["status"] == "success":
            final_metrics = final_result["metrics"]
            final_predictions = pd.DataFrame(
                {
                    "timestamp_input": final_result["timestamps"],
                    "timestamp_target": final_result["timestamps"],
                    "actual_DO": final_result["y_true"],
                    "predicted_DO": final_result["predictions"],
                    "absolute_error": abs(final_result["y_true"] - final_result["predictions"]),
                    "squared_error": (final_result["y_true"] - final_result["predictions"]) ** 2,
                    "forecast_horizon_minutes": horizon.median_interval_minutes * horizon.horizon_steps,
                    "feature_set": "rfe_cfs",
                    "model": best_model,
                }
            )
            final_predictions.to_csv(CHAPTER5_RESULTS / "predictions" / "do_10min_predictions.csv", index=False)
            alerts = _generate_alerts(final_predictions)
            alerts.to_csv(CHAPTER5_RESULTS / "predictions" / "do_prediction_alerts.csv", index=False)
            plot_actual_vs_predicted(
                final_predictions["timestamp_input"],
                final_predictions["actual_DO"],
                final_predictions["predicted_DO"],
                f"Final Model ({best_model}) DO Forecast",
                CHAPTER5_RESULTS / "figures" / "final_actual_vs_predicted",
            )

    if not comparison.empty:
        for feature_set in active_sets:
            subset = comparison[comparison["Feature_Set"] == feature_set]
            if not subset.empty:
                plot_metric_comparison(
                    subset,
                    "RMSE",
                    f"RMSE Comparison ({feature_set})",
                    CHAPTER5_RESULTS / "figures" / f"rmse_{feature_set}",
                )

    if failures:
        pd.DataFrame(failures).to_csv(CHAPTER5_RESULTS / "metrics" / "model_failures.csv", index=False)

    def _best_for_set(name: str) -> dict[str, float] | None:
        sub = comparison[comparison["Feature_Set"] == name]
        if sub.empty:
            return None
        row = sub.sort_values("RMSE").iloc[0]
        return {"Model": row["Model"], "RMSE": row["RMSE"], "MAE": row["MAE"], "R2": row["R2"]}

    summary.update(
        {
            "status": "success" if not comparison.empty else "failed",
            "fast_mode": FAST_MODE,
            "dataset_rows": len(raw),
            "train_size": len(train),
            "validation_size": len(val),
            "test_size": len(test),
            "horizon": horizon.to_dict(),
            "candidate_feature_count": len(all_features),
            "rfe_feature_count": len(rfe_result.selected_features),
            "cfs_feature_count": len(cfs_result.selected_features),
            "rfe_cfs_feature_count": len(combined_result.selected_features),
            "evaluated_feature_sets": list(active_sets.keys()),
            "final_selected_features": best_features,
            "final_proposed_model": best_model,
            "final_metrics": final_metrics,
            "best_all_features": _best_for_set("all_features"),
            "best_rfe": _best_for_set("rfe"),
            "best_cfs": _best_for_set("cfs"),
            "best_rfe_cfs": _best_for_set("rfe_cfs"),
            "successful_models": comparison[["Feature_Set", "Model"]].drop_duplicates().to_dict(orient="records")
            if not comparison.empty
            else [],
            "failed_models": failures,
            **periods,
        }
    )

    write_json(CHAPTER5_RESULTS / "metrics" / "chapter5_summary.json", summary)
    write_excel_sheets(
        CHAPTER5_RESULTS / "tables" / "chapter5_results.xlsx",
        {
            "Chapter5_All_Features": comparison[comparison["Feature_Set"] == "all_features"],
            "Chapter5_RFE": comparison[comparison["Feature_Set"] == "rfe"],
            "Chapter5_CFS": comparison[comparison["Feature_Set"] == "cfs"],
            "Chapter5_RFE_CFS": comparison[comparison["Feature_Set"] == "rfe_cfs"],
            "Chapter5_Model_Comparison": comparison,
            "Chapter5_Predictions": final_predictions if final_predictions is not None else pd.DataFrame(),
            "Chapter5_Features": pd.DataFrame({"feature_name": best_features}),
        },
    )

    logger.info("Chapter 5 complete. Final model: %s", best_model)
    return summary


if __name__ == "__main__":
    result = run_chapter5()
    print(json.dumps(result, indent=2, default=str))
