"""Load backend result files — read-only, no training."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
CHAPTER3_RESULTS = RESULTS_ROOT / "chapter3_ph"
CHAPTER5_RESULTS = RESULTS_ROOT / "chapter5_do"


def file_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError):
        return None


def latest_prediction_file(results_dir: Path, pattern: str) -> Path | None:
    files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def chapter3_bundle() -> dict:
    summary_path = CHAPTER3_RESULTS / "metrics" / "chapter3_summary.json"
    comparison = load_csv(CHAPTER3_RESULTS / "metrics" / "chapter3_model_comparison.csv")
    failures = load_csv(CHAPTER3_RESULTS / "metrics" / "model_failures.csv")
    load_report = load_json(CHAPTER3_RESULTS / "data" / "load_report.json")
    prep = load_csv(CHAPTER3_RESULTS / "data" / "preprocessing_summary.csv")

    pred_dir = CHAPTER3_RESULTS / "predictions"
    pred_files = {p.stem.replace("_predictions", ""): p for p in pred_dir.glob("*_predictions.csv")}
    model_predictions = {name: load_csv(path) for name, path in pred_files.items()}

    hybrid_preds = model_predictions.get("arima_lstm_hybrid")
    if hybrid_preds is None:
        hybrid_preds = load_csv(pred_dir / "hybrid_predictions.csv")

    hybrid_summary = load_json(CHAPTER3_RESULTS / "predictions" / "hybrid_model_summary.json")
    if hybrid_summary is None:
        hybrid_summary = load_json(CHAPTER3_RESULTS / "metrics" / "hybrid_model_summary.json")

    summary = load_json(summary_path)

    return {
        "summary": summary,
        "summary_path": summary_path,
        "comparison": comparison,
        "failures": failures,
        "load_report": load_report,
        "preprocessing": prep,
        "hybrid_predictions": hybrid_preds,
        "hybrid_summary": hybrid_summary,
        "model_predictions": model_predictions,
        "prediction_files": pred_files,
        "last_run": file_mtime(summary_path) or file_mtime(CHAPTER3_RESULTS / "logs" / "chapter3.log"),
    }


def chapter5_bundle() -> dict:
    summary_path = CHAPTER5_RESULTS / "metrics" / "chapter5_summary.json"
    comparison = load_csv(CHAPTER5_RESULTS / "metrics" / "chapter5_feature_selection_model_comparison.csv")
    failures = load_csv(CHAPTER5_RESULTS / "metrics" / "model_failures.csv")
    preds = load_csv(CHAPTER5_RESULTS / "predictions" / "do_10min_predictions.csv")
    alerts = load_csv(CHAPTER5_RESULTS / "predictions" / "do_prediction_alerts.csv")
    load_report = load_json(CHAPTER5_RESULTS / "data" / "load_report.json")
    horizon = load_json(CHAPTER5_RESULTS / "data" / "horizon_report.json")
    feat_dict = load_csv(CHAPTER5_RESULTS / "data" / "do_feature_dictionary.csv")
    fs_dir = CHAPTER5_RESULTS / "feature_selection"
    fs_report = load_json(fs_dir / "final_feature_selection_report.json")
    rfe_features = load_csv(fs_dir / "rfe_selected_features.csv")
    cfs_features = load_csv(fs_dir / "cfs_selected_features.csv")
    rfe_cfs_features = load_csv(fs_dir / "final_rfe_cfs_features.csv")
    rfe_val_scores = load_csv(fs_dir / "rfe_validation_scores.csv")
    summary = load_json(summary_path)

    return {
        "summary": summary,
        "summary_path": summary_path,
        "comparison": comparison,
        "failures": failures,
        "predictions": preds,
        "alerts": alerts,
        "load_report": load_report,
        "horizon": horizon,
        "feature_dictionary": feat_dict,
        "feature_selection_report": fs_report,
        "rfe_features": rfe_features,
        "cfs_features": cfs_features,
        "rfe_cfs_features": rfe_cfs_features,
        "rfe_validation_scores": rfe_val_scores,
        "last_run": file_mtime(summary_path) or file_mtime(CHAPTER5_RESULTS / "logs" / "chapter5.log"),
    }
