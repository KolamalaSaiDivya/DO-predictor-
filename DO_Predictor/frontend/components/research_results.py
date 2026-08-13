"""Cached research result loading and traceability helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
CHAPTER3_RESULTS = RESULTS_ROOT / "chapter3_ph"
CHAPTER5_RESULTS = RESULTS_ROOT / "chapter5_do"

MODEL_LABELS = {
    "persistence": "Persistence",
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "svr": "SVR",
    "arima": "ARIMA",
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
    "arima_lstm_hybrid": "ARLSTMIMA",
    "xgboost": "XGBoost",
}

FS_LABELS = {
    "all_features": "Full Feature Set",
    "rfe": "RFE Feature Set",
    "cfs": "CFS Feature Set",
    "rfe_cfs": "RFE+CFS Feature Set",
}


def _mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _csv(path: Path, nrows: int | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, nrows=nrows)
    except (pd.errors.ParserError, OSError):
        return None


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def fs_label(name: str) -> str:
    return FS_LABELS.get(name, name.replace("_", " ").title())


def clear_result_caches() -> None:
    """Invalidate cached result loaders (called by Refresh Results)."""
    load_chapter3.clear()
    load_chapter5.clear()


@st.cache_data(show_spinner=False)
def load_chapter3() -> dict:
    summary_path = CHAPTER3_RESULTS / "metrics" / "chapter3_summary.json"
    pred_dir = CHAPTER3_RESULTS / "predictions"
    pred_files = {p.stem.replace("_predictions", ""): p for p in pred_dir.glob("*_predictions.csv")}
    model_predictions = {n: _csv(p) for n, p in pred_files.items()}

    return {
        "summary": _json(summary_path),
        "summary_path": str(summary_path),
        "comparison": _csv(CHAPTER3_RESULTS / "metrics" / "chapter3_model_comparison.csv"),
        "ranking": _csv(CHAPTER3_RESULTS / "metrics" / "chapter3_model_ranking.csv"),
        "failures": _csv(CHAPTER3_RESULTS / "metrics" / "model_failures.csv"),
        "load_report": _json(CHAPTER3_RESULTS / "data" / "load_report.json"),
        "preprocessing": _csv(CHAPTER3_RESULTS / "data" / "preprocessing_summary.csv"),
        "cleaned_sample": _csv(CHAPTER3_RESULTS / "data" / "cleaned_dataset.csv", nrows=10),
        "hybrid_components": _csv(pred_dir / "hybrid_components.csv"),
        "hybrid_summary": _json(CHAPTER3_RESULTS / "metrics" / "hybrid_model_summary.json"),
        "model_predictions": model_predictions,
        "last_run": _mtime(summary_path),
        "data_path": str(PROJECT_ROOT / "data" / "raw" / "chapter3_ph_dataset.csv"),
    }


@st.cache_data(show_spinner=False)
def load_chapter5() -> dict:
    summary_path = CHAPTER5_RESULTS / "metrics" / "chapter5_summary.json"
    fs_dir = CHAPTER5_RESULTS / "feature_selection"
    return {
        "summary": _json(summary_path),
        "summary_path": str(summary_path),
        "comparison": _csv(CHAPTER5_RESULTS / "metrics" / "chapter5_feature_selection_model_comparison.csv"),
        "failures": _csv(CHAPTER5_RESULTS / "metrics" / "model_failures.csv"),
        "predictions": _csv(CHAPTER5_RESULTS / "predictions" / "do_10min_predictions.csv"),
        "alerts": _csv(CHAPTER5_RESULTS / "predictions" / "do_prediction_alerts.csv"),
        "load_report": _json(CHAPTER5_RESULTS / "data" / "load_report.json"),
        "preprocessing": _csv(CHAPTER5_RESULTS / "data" / "preprocessing_summary.csv"),
        "cleaned_sample": _csv(CHAPTER5_RESULTS / "data" / "cleaned_dataset.csv", nrows=10),
        "horizon": _json(CHAPTER5_RESULTS / "data" / "horizon_report.json"),
        "feature_dictionary": _csv(CHAPTER5_RESULTS / "data" / "do_feature_dictionary.csv"),
        "fs_report": _json(fs_dir / "final_feature_selection_report.json"),
        "rfe_features": _csv(fs_dir / "rfe_selected_features.csv"),
        "rfe_rankings": _csv(fs_dir / "rfe_rankings.csv"),
        "rfe_val_scores": _csv(fs_dir / "rfe_validation_scores.csv"),
        "cfs_features": _csv(fs_dir / "cfs_selected_features.csv"),
        "cfs_scores": _csv(fs_dir / "cfs_scores.csv"),
        "rfe_cfs_features": _csv(fs_dir / "final_rfe_cfs_features.csv"),
        "last_run": _mtime(summary_path),
        "data_path": str(PROJECT_ROOT / "data" / "raw" / "chapter5_do_dataset.csv"),
    }


def model_metrics(comparison: pd.DataFrame | None, model_key: str) -> dict | None:
    if comparison is None or comparison.empty:
        return None
    row = comparison[comparison["Model"] == model_key]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def model_failed(failures: pd.DataFrame | None, model_key: str) -> str | None:
    if failures is None or failures.empty:
        return None
    col = "Model" if "Model" in failures.columns else failures.columns[0]
    row = failures[failures[col] == model_key]
    if row.empty:
        return None
    err_col = "error" if "error" in failures.columns else failures.columns[-1]
    return str(row.iloc[0][err_col])


def model_status(summary: dict | None, failures: pd.DataFrame | None, model_key: str) -> str:
    if summary is None:
        return "unavailable"
    ok = summary.get("successful_models") or []
    fail = summary.get("failed_models") or []
    if model_key in ok:
        return "completed"
    if model_key in fail:
        return "failed"
    if model_failed(failures, model_key):
        return "failed"
    return "not_run"


def prep_target_stats(prep: pd.DataFrame | None, target_col: str) -> dict | None:
    if prep is None or prep.empty or "column" not in prep.columns:
        return None
    row = prep[prep["column"] == target_col]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "min": r.get("min"),
        "max": r.get("max"),
        "mean": r.get("mean"),
        "std": r.get("std"),
        "missing_pct": r.get("missing_percentage"),
    }


def best_model_row(comparison: pd.DataFrame | None) -> dict | None:
    if comparison is None or comparison.empty or "RMSE" not in comparison.columns:
        return None
    return comparison.sort_values("RMSE").iloc[0].to_dict()


def feature_type_counts(fdict: pd.DataFrame | None) -> dict:
    if fdict is None or fdict.empty or "feature_type" not in fdict.columns:
        return {}
    counts = fdict["feature_type"].astype(str).value_counts().to_dict()
    rolling = int(fdict["feature_type"].astype(str).str.contains("rolling").sum())
    return {"lag": counts.get("lag", 0), "rolling": rolling, "total": len(fdict)}
