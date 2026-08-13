"""Chapter 3 experiment: multivariate pH forecasting with ARIMA-LSTM hybrid."""

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
    CHAPTER3_DATA_PATH,
    CHAPTER3_RESULTS,
    CHAPTER3_TARGET,
    EXCLUDE_FROM_FEATURES,
    FAST_MODE,
    MIN_FEATURE_COVERAGE,
    TIMESTAMP_COLUMN_CHAPTER3,
    effective_chapter3_models,
    effective_lag_steps,
    effective_max_rows,
    effective_rolling_windows,
)
from src.data_quality import preprocess_dataset
from src.logging_utils import setup_experiment_logger
from src.plotting import (
    plot_actual_vs_predicted,
    plot_loss_history,
    plot_metric_comparison,
    plot_residuals,
    plot_time_series,
    plot_train_val_test_split,
)
from src.preprocessing import chronological_split, drop_incomplete_feature_rows, drop_na_features, filter_features_by_coverage, split_period_strings, subsample_for_fast_mode
from src.reproducibility import set_global_seeds
from src.research_data_loader import get_numeric_sensor_columns, load_research_dataset, save_load_report
from src.research_evaluation import rank_models
from src.research_feature_engineering import add_forecast_target, create_temporal_features, list_feature_columns
from src.research_model_runner import safe_run_model
from src.results_export import save_dataframe, write_excel_sheets, write_json

logger = setup_experiment_logger(__name__, CHAPTER3_RESULTS / "logs" / "chapter3.log")


def run_chapter3() -> dict[str, Any]:
    set_global_seeds()
    summary: dict[str, Any] = {"status": "failed", "chapter": "chapter3_ph"}

    if not CHAPTER3_DATA_PATH.exists():
        msg = (
            f"Chapter 3 dataset not found at {CHAPTER3_DATA_PATH}. "
            "Place chapter3_ph_dataset.csv at this path before running Chapter 3."
        )
        logger.error(msg)
        summary["error"] = msg
        return summary

    raw, load_report = load_research_dataset(CHAPTER3_DATA_PATH, timestamp_override=TIMESTAMP_COLUMN_CHAPTER3)
    save_load_report(load_report, CHAPTER3_RESULTS / "data" / "load_report.json")

    target_col = CHAPTER3_TARGET
    if target_col not in raw.columns:
        summary["error"] = f"Target column '{target_col}' not found in Chapter 3 dataset."
        logger.error(summary["error"])
        return summary

    cleaned, prep_summary, prep_report = preprocess_dataset(raw, target_col)
    prep_summary.to_csv(CHAPTER3_RESULTS / "data" / "preprocessing_summary.csv", index=False)
    cleaned.to_csv(CHAPTER3_RESULTS / "data" / "cleaned_dataset.csv", index=False)
    cleaned = subsample_for_fast_mode(cleaned, effective_max_rows())

    sensor_cols = get_numeric_sensor_columns(cleaned, target_col)
    featured, _ = create_temporal_features(
        cleaned,
        sensor_cols,
        lags=effective_lag_steps(),
        rolling_windows=effective_rolling_windows(),
    )
    featured = add_forecast_target(featured, target_col, horizon_steps=1)
    featured = drop_na_features(featured, target_col)
    if FAST_MODE:
        logger.info("FAST_MODE active — results are for smoke testing only.")

    feature_cols = list_feature_columns(featured, EXCLUDE_FROM_FEATURES + [target_col, f"{target_col}_target"])
    feature_cols = filter_features_by_coverage(featured, feature_cols, MIN_FEATURE_COVERAGE)
    featured = drop_incomplete_feature_rows(featured, feature_cols, target_col)
    if featured.empty:
        summary["error"] = "No rows remain after removing incomplete feature rows."
        logger.error(summary["error"])
        return summary
    train, val, test = chronological_split(featured)
    periods = split_period_strings(train, val, test)

    plot_time_series(cleaned, "Timestamp", target_col, "pH Time Series", CHAPTER3_RESULTS / "figures" / "ph_timeseries")
    plot_train_val_test_split(
        featured,
        "Timestamp",
        target_col,
        (len(train), len(train) + len(val)),
        CHAPTER3_RESULTS / "figures" / "train_val_test_split",
    )

    comparison_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    predictions_store: dict[str, pd.DataFrame] = {}

    for model_name in effective_chapter3_models():
        logger.info("Running Chapter 3 model: %s", model_name)
        result = safe_run_model(model_name, train, val, test, target_col, feature_cols)
        if result["status"] != "success":
            failures.append({"Model": model_name, "error": result.get("error", "unknown")})
            continue
        pred_df = pd.DataFrame(
            {
                "Timestamp": result["timestamps"],
                "actual": result["y_true"],
                "predicted": result["predictions"],
            }
        )
        if pred_df.empty:
            failures.append({"Model": model_name, "error": "No test predictions produced."})
            continue
        metrics = result["metrics"]
        comparison_rows.append(
            {
                "Model": model_name,
                "Target": target_col,
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "R2": metrics["R2"],
                "Training_Time": result["training_time"],
                "Prediction_Time": result["prediction_time"],
            }
        )
        predictions_store[model_name] = pred_df
        pred_df.to_csv(CHAPTER3_RESULTS / "predictions" / f"{model_name}_predictions.csv", index=False)

        if model_name == "arima_lstm_hybrid" and result.get("components"):
            comp = result["components"]
            pd.DataFrame(
                {
                    "Timestamp": result["timestamps"],
                    "actual": result["y_true"],
                    "arima_prediction": comp.get("arima"),
                    "lstm_residual_prediction": comp.get("lstm_residual"),
                    "hybrid_prediction": result["predictions"],
                }
            ).to_csv(CHAPTER3_RESULTS / "predictions" / "hybrid_components.csv", index=False)
            hybrid_meta = {
                "algorithm": "y_hat_hybrid = y_hat_ARIMA + e_hat_LSTM",
                "arima_summary": result.get("arima_summary"),
                "training_time_seconds": result.get("training_time"),
                "prediction_time_seconds": result.get("prediction_time"),
                "metrics": metrics,
            }
            write_json(CHAPTER3_RESULTS / "metrics" / "hybrid_model_summary.json", hybrid_meta)

        if not result.get("history_df", pd.DataFrame()).empty:
            plot_loss_history(
                result["history_df"],
                f"{model_name} Training History",
                CHAPTER3_RESULTS / "figures" / f"{model_name}_loss",
            )
            result["history_df"].to_csv(
                CHAPTER3_RESULTS / "metrics" / f"{model_name}_training_history.csv", index=False
            )

        plot_actual_vs_predicted(
            pred_df["Timestamp"],
            pred_df["actual"],
            pred_df["predicted"],
            f"{model_name} Actual vs Predicted pH",
            CHAPTER3_RESULTS / "figures" / f"{model_name}_actual_vs_predicted",
        )
        plot_residuals(
            pred_df["actual"].to_numpy(),
            pred_df["predicted"].to_numpy(),
            f"{model_name} Residual Distribution",
            CHAPTER3_RESULTS / "figures" / f"{model_name}_residuals",
        )

    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        comparison.to_csv(CHAPTER3_RESULTS / "metrics" / "chapter3_model_comparison.csv", index=False)
        ranking = rank_models(comparison)
        ranking.to_csv(CHAPTER3_RESULTS / "metrics" / "chapter3_model_ranking.csv", index=False)
        for metric in ("MAE", "RMSE", "R2"):
            plot_metric_comparison(
                comparison,
                metric,
                f"Chapter 3 {metric} Comparison",
                CHAPTER3_RESULTS / "figures" / f"comparison_{metric.lower()}",
            )

    if failures:
        pd.DataFrame(failures).to_csv(CHAPTER3_RESULTS / "metrics" / "model_failures.csv", index=False)

    best_model = comparison.sort_values("RMSE").iloc[0]["Model"] if not comparison.empty else None
    summary.update(
        {
            "status": "success" if not comparison.empty else "failed",
            "fast_mode": FAST_MODE,
            "dataset_rows": len(cleaned),
            "dataset_rows_raw": len(raw),
            "train_size": len(train),
            "validation_size": len(val),
            "test_size": len(test),
            "best_model": best_model,
            "successful_models": comparison["Model"].tolist() if not comparison.empty else [],
            "failed_models": [f["Model"] for f in failures],
            **periods,
        }
    )

    write_json(CHAPTER3_RESULTS / "metrics" / "chapter3_summary.json", summary)
    write_excel_sheets(
        CHAPTER3_RESULTS / "tables" / "chapter3_results.xlsx",
        {
            "Chapter3_Model_Comparison": comparison,
            "Chapter3_Predictions": pd.concat(
                [df.assign(Model=name) for name, df in predictions_store.items()], ignore_index=True
            )
            if predictions_store
            else pd.DataFrame(),
            "Chapter3_Metrics": comparison,
        },
    )

    logger.info("Chapter 3 complete. Successful models: %s", summary.get("successful_models"))
    logger.info("Failed models: %s", summary.get("failed_models"))
    return summary


if __name__ == "__main__":
    result = run_chapter3()
    print(json.dumps(result, indent=2, default=str))
