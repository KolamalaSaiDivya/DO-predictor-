"""Two things happen here, both Dissolved-Oxygen-only:

1. Preprocessing-stage ablation: a small representative subset of models
   (one from each family) run against all four pipeline stages, to isolate
   how much cleaning / smoothing / feature engineering actually buys you.
   Running the *entire* registry at every stage would multiply an already
   expensive DL/SARIMAX/Prophet training bill by 4 for no real benefit - the
   preprocessing question doesn't need 20 models to answer, it needs a few
   models spanning statistical/classical/deep learning.
2. Full model comparison: every model in MODEL_REGISTRY on the final
   (full_features) stage only - this is the actual thesis benchmark table,
   compared against the persistence floor and the cited SARIMAX/Prophet
   public baseline.

Run with `python -m experiments.run_ablation` from the project root.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    BASELINE_PROPHET_MAE,
    BASELINE_SARIMAX_MAE,
    PRIMARY_TARGET,
    RESULTS_DIR,
    get_logger,
)
from src.evaluation import StageMetricLogger, compare_to_baselines
from src.models.registry import MODEL_REGISTRY
from src.pipeline import STAGE_NAMES, build_base_artifacts, build_stage_dataframe, train_and_evaluate

logger = get_logger(__name__)

REPRESENTATIVE_STAGE_MODELS = [
    "persistence",
    "linear_regression",
    "random_forest",
    "xgboost",
    "lstm",
]


def run_stage_ablation(metric_logger: StageMetricLogger) -> None:
    base = build_base_artifacts()
    models_to_run = [m for m in REPRESENTATIVE_STAGE_MODELS if m in MODEL_REGISTRY]
    logger.info("Stage ablation using representative models: %s", models_to_run)

    for stage in STAGE_NAMES:
        stage_df = build_stage_dataframe(base, PRIMARY_TARGET, stage)
        for model_name in models_to_run:
            t0 = time.time()
            try:
                train_and_evaluate(model_name, PRIMARY_TARGET, stage_df, stage, metric_logger)
            except Exception:
                logger.exception("Stage ablation failed for %s/%s - skipping.", stage, model_name)
                continue
            logger.info("done %s/%s in %.1fs", stage, model_name, time.time() - t0)


def run_full_comparison(metric_logger: StageMetricLogger) -> list[dict]:
    base = build_base_artifacts()
    comparison_rows: list[dict] = []

    stage_df = build_stage_dataframe(base, PRIMARY_TARGET, "full_features")
    for model_name in MODEL_REGISTRY:
        t0 = time.time()
        try:
            result = train_and_evaluate(model_name, PRIMARY_TARGET, stage_df, "full_features", metric_logger)
        except Exception:
            logger.exception("Full comparison failed for %s - skipping, NOT silently dropped from report.", model_name)
            comparison_rows.append({"target": PRIMARY_TARGET, "model": model_name, "status": "failed"})
            continue
        elapsed = time.time() - t0
        if result is None:
            comparison_rows.append({"target": PRIMARY_TARGET, "model": model_name, "status": "insufficient_data"})
            continue

        row = {
            "target": PRIMARY_TARGET,
            "model": model_name,
            "status": "ok",
            "train_time_seconds": round(elapsed, 2),
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            **{f"test_{k}": v for k, v in result["test_metrics"].items()},
        }
        row.update(compare_to_baselines(
            result["test_metrics"]["mae"],
            {"sarimax_baseline": BASELINE_SARIMAX_MAE, "prophet_baseline": BASELINE_PROPHET_MAE},
        ))
        comparison_rows.append(row)
        logger.info("done %s in %.1fs -> MAE=%.4f", model_name, elapsed, row.get("test_mae", float("nan")))

    return comparison_rows


def main() -> None:
    metric_logger = StageMetricLogger()

    logger.info("=== Stage ablation ===")
    run_stage_ablation(metric_logger)

    stage_metrics_path = RESULTS_DIR / "stage_metrics.json"
    stage_metrics_path.write_text(json.dumps(metric_logger.to_dict_list(), indent=2, default=str))
    logger.info("Wrote %s", stage_metrics_path)

    logger.info("=== Full model comparison ===")
    comparison_metric_logger = StageMetricLogger()
    comparison_rows = run_full_comparison(comparison_metric_logger)

    comparison_path = RESULTS_DIR / "model_comparison.json"
    comparison_path.write_text(json.dumps(comparison_rows, indent=2, default=str))
    logger.info("Wrote %s", comparison_path)

    n_ok = sum(1 for r in comparison_rows if r["status"] == "ok")
    n_failed = sum(1 for r in comparison_rows if r["status"] == "failed")
    logger.info("Full comparison: %d ok, %d failed, %d total.", n_ok, n_failed, len(comparison_rows))


if __name__ == "__main__":
    main()
