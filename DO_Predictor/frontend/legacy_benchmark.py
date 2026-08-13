"""Legacy DO benchmark (train-from-Streamlit). Run: streamlit run frontend/legacy_benchmark.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    BASELINE_PROPHET_MAE,
    BASELINE_SARIMAX_MAE,
    LOW_DO_THRESHOLD_MGL,
    PRIMARY_TARGET,
    RAW_CSV_PATH,
    SENSOR_COLUMNS,
)
from src.evaluation import StageMetricLogger
from src.models.registry import CLASSICAL_ML_MODELS, DEEP_LEARNING_MODELS, MODEL_REGISTRY, STATISTICAL_MODELS
from src.pipeline import build_base_artifacts, build_stage_dataframe, train_and_evaluate

st.set_page_config(page_title="DO_Predictor Legacy Benchmark", layout="wide")

QUICK_MODELS = ["persistence", "linear_regression", "ridge", "random_forest", "xgboost", "lstm"]


@st.cache_resource(show_spinner="Loading and cleaning dataset...")
def get_base_artifacts(csv_path_str: str | None):
    csv_path = Path(csv_path_str) if csv_path_str else None
    return build_base_artifacts(csv_path=csv_path)


def run_models(base, model_names: list[str], stage: str) -> pd.DataFrame:
    stage_df = build_stage_dataframe(base, PRIMARY_TARGET, stage)
    metric_logger = StageMetricLogger()
    rows = []
    progress = st.progress(0.0, text="Training models...")
    for i, name in enumerate(model_names):
        progress.progress((i + 1) / len(model_names), text=f"Training {name}...")
        try:
            result = train_and_evaluate(name, PRIMARY_TARGET, stage_df, stage, metric_logger)
        except Exception as exc:
            rows.append({"model": name, "status": "failed", "error": str(exc)})
            continue
        if result is None:
            rows.append({"model": name, "status": "insufficient_data"})
            continue
        m = result["test_metrics"]
        rows.append({
            "model": name, "status": "ok",
            "mae": m["mae"], "rmse": m["rmse"], "mape": m["mape"], "r2": m["r2"],
            "n_test": result["n_test"],
        })
        st.session_state.setdefault("fitted_models", {})[name] = result["model"]
    progress.empty()
    return pd.DataFrame(rows)


def category_of(model_name: str) -> str:
    if model_name in STATISTICAL_MODELS:
        return "Statistical"
    if model_name in CLASSICAL_ML_MODELS:
        return "Classical ML"
    if model_name in DEEP_LEARNING_MODELS:
        return "Deep Learning"
    return "Unknown"


st.title("DO_Predictor - Legacy DO Benchmark")
st.caption("Original 20-model training demo. For thesis results use: streamlit run frontend/streamlit_app.py")

with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload CSV (optional)", type="csv")
    csv_path_str: str | None = None
    if uploaded is not None:
        tmp_path = Path("data/raw/streamlit_upload.csv")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.getvalue())
        csv_path_str = str(tmp_path)
    else:
        st.info(f"Using: {RAW_CSV_PATH.name}")

    st.header("2. Models")
    mode = st.radio("Model set", ["Quick (6 models)", "Full registry"])
    model_names = QUICK_MODELS if mode.startswith("Quick") else list(MODEL_REGISTRY)
    stage = st.selectbox("Pipeline stage", ["full_features", "lag_features", "smoothed", "cleaned"])
    run_button = st.button("Run pipeline & train", type="primary")

base = get_base_artifacts(csv_path_str)
tab_quality, tab_eda, tab_results = st.tabs(["Data Quality", "EDA", "Model Comparison"])

with tab_quality:
    qr = base.quality_report
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", qr.n_rows)
    c2.metric("Duplicate timestamps", qr.duplicate_timestamp_count)
    c3.metric("Modal cadence (min)", qr.modal_cadence_minutes)
    c4.metric("Quality cols kept / dropped", f"{len(qr.quality_columns_kept)} / {len(qr.quality_columns_dropped)}")
    for note in qr.notes:
        st.markdown(f"- {note}")

with tab_eda:
    st.plotly_chart(
        px.line(base.cleaned, x="Timestamp", y=PRIMARY_TARGET, title=f"{PRIMARY_TARGET} over time"),
        use_container_width=True,
    )

with tab_results:
    if run_button:
        st.session_state["results_df"] = run_models(base, model_names, stage)
    results_df = st.session_state.get("results_df")
    if results_df is None:
        st.info("Click **Run pipeline & train** in the sidebar.")
    else:
        ok_df = results_df[results_df["status"] == "ok"].copy().sort_values("mae")
        if ok_df.empty:
            st.dataframe(results_df)
        else:
            ok_df["category"] = ok_df["model"].apply(category_of)
            st.dataframe(ok_df[["model", "category", "mae", "rmse", "mape", "r2", "n_test"]], use_container_width=True)
            fig = go.Figure()
            fig.add_bar(x=ok_df["model"], y=ok_df["mae"], marker_color="#2563eb")
            fig.add_hline(y=BASELINE_SARIMAX_MAE, line_dash="dash", line_color="#d97706")
            fig.add_hline(y=BASELINE_PROPHET_MAE, line_dash="dash", line_color="#dc2626")
            st.plotly_chart(fig, use_container_width=True)
