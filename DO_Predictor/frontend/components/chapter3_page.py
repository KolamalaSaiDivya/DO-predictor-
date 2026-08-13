"""Chapter 3 interactive process page — real backend results only."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from config.research_config import LOOKBACK
from frontend.components import pipeline_ui as ui
from frontend.components.research_results import (
    best_model_row,
    label,
    load_chapter3,
    model_failed,
    model_metrics,
    model_status,
    prep_target_stats,
)

CH3_STAGES = [
    "data",
    "preprocessing",
    "arima",
    "lstm",
    "hybrid",
    "arlstima",
    "ph_prediction",
    "model_comparison",
    "final_result",
]

HYBRID_DESC = (
    "Residual hybrid: ARIMA fits the pH series; LSTM predicts ARIMA residuals using "
    "engineered features plus arima_fitted and arima_residual_lag1. "
    "Final prediction = ARIMA forecast + LSTM residual correction."
)


def _stages(data: dict) -> list[tuple[str, str, str]]:
    s, f = data["summary"], data["failures"]
    return [
        ("data", "DATA", "completed" if data["load_report"] else "failed"),
        ("preprocessing", "PREPROCESSING", "completed" if data["preprocessing"] is not None else "not_run"),
        ("arima", "ARIMA", model_status(s, f, "arima")),
        ("lstm", "LSTM", model_status(s, f, "lstm")),
        ("hybrid", "ARIMA + LSTM HYBRID", model_status(s, f, "arima_lstm_hybrid")),
        ("arlstima", "ARLSTMIMA", model_status(s, f, "arima_lstm_hybrid")),
        ("ph_prediction", "pH PREDICTION", "completed" if data["model_predictions"] else "not_run"),
        ("model_comparison", "MODEL COMPARISON", "completed" if data["comparison"] is not None and not data["comparison"].empty else "not_run"),
        ("final_result", "FINAL RESULT", "completed" if s and s.get("status") == "success" else "failed"),
    ]


def _main_result(data: dict) -> tuple[dict | None, pd.DataFrame | None]:
    summary = data["summary"] or {}
    best_key = summary.get("best_model")
    preds = data["model_predictions"].get(best_key) if best_key else None
    if preds is None or preds.empty:
        for k, df in data["model_predictions"].items():
            if df is not None and not df.empty:
                best_key, preds = k, df
                break
    metrics = model_metrics(data["comparison"], best_key) if best_key else None
    return metrics, preds


def _render_data(data: dict) -> None:
    ui.inline_stage_intro("AIoT Water Quality Data", "Load and validate the Chapter 3 pH dataset.", "completed" if data["load_report"] else "failed")
    lr = data["load_report"] or {}
    ui.kv_table(
        [
            ("Dataset file", lr.get("filename", "chapter3_ph_dataset.csv")),
            ("Source path", data["data_path"]),
            ("Total rows", str(lr.get("n_rows", "—"))),
            ("Total columns", str(lr.get("n_columns", "—"))),
            ("Target column", "pH"),
            ("Timestamp column", "Timestamp"),
            ("Duplicate rows", str(lr.get("duplicate_rows", "—"))),
            ("Duplicate timestamps", str(lr.get("duplicate_timestamps", "—"))),
            ("Time range", " → ".join(lr.get("time_range", [])) if lr.get("time_range") else "—"),
        ]
    )
    ui.section("Data Quality")
    miss = lr.get("missing_counts") or {}
    if miss:
        miss_df = pd.DataFrame([{"Column": k, "Missing": v} for k, v in miss.items()])
        st.dataframe(miss_df.head(12), use_container_width=True, hide_index=True)
    ui.section("Data Sample")
    if data["cleaned_sample"] is not None and not data["cleaned_sample"].empty:
        st.dataframe(data["cleaned_sample"].head(10), use_container_width=True, hide_index=True)
    else:
        st.caption("Not available from current run.")
    ui.section("Target Statistics (pH)")
    stats = prep_target_stats(data["preprocessing"], "pH")
    if stats:
        ui.kv_table([(k, str(v)) for k, v in stats.items()])
    ui.section("Visualization")
    if data["cleaned_sample"] is not None and "pH" in data["cleaned_sample"].columns:
        ui.line_chart(data["cleaned_sample"], "Timestamp" if "Timestamp" in data["cleaned_sample"].columns else data["cleaned_sample"].columns[0], ["pH"], ["pH"], "pH", "pH in loaded sample")
    ui.section("Next Stage")
    st.write("→ PREPROCESSING")


def _render_preprocessing(data: dict) -> None:
    ui.inline_stage_intro("Data Preprocessing", "Clean, validate, order, and split data for forecasting.", "completed" if data["preprocessing"] is not None else "not_run")
    s = data["summary"] or {}
    ui.io_flow(
        f"{s.get('dataset_rows_raw', '—')} raw observations",
        "Missing-value handling · Chronological ordering · Feature alignment · Train/val/test split",
        f"{s.get('dataset_rows', '—')} processed rows · train {s.get('train_size', '—')} · val {s.get('validation_size', '—')} · test {s.get('test_size', '—')}",
    )
    ui.section("Preprocessing Summary")
    if data["preprocessing"] is not None:
        st.dataframe(data["preprocessing"], use_container_width=True, hide_index=True)
    ui.section("Validation")
    ui.kv_table(
        [
            ("Target column present", "pH"),
            ("Chronological split", "Yes"),
            ("Train period", s.get("training_period", "—")),
            ("Validation period", s.get("validation_period", "—")),
            ("Test period", s.get("test_period", "—")),
            ("Sequence window (LOOKBACK)", str(LOOKBACK)),
        ]
    )
    ui.section("Next Stage")
    st.write("→ ARIMA")


def _render_model_stage(data: dict, model_key: str, title: str, purpose: str) -> None:
    status = model_status(data["summary"], data["failures"], model_key)
    ui.inline_stage_intro(title, purpose, status)
    if status == "failed":
        err = model_failed(data["failures"], model_key)
        st.error(err or "Model failed during execution.")
        return
    if status != "completed":
        st.caption("Not available from current run.")
        return
    metrics = model_metrics(data["comparison"], model_key)
    preds = data["model_predictions"].get(model_key)
    s = data["summary"] or {}
    ui.section("Input")
    ui.kv_table(
        [
            ("Training samples", str(s.get("train_size", "—"))),
            ("Validation samples", str(s.get("validation_size", "—"))),
            ("Test samples", str(s.get("test_size", "—"))),
            ("Target", "pH"),
        ]
    )
    ui.section("Output")
    if preds is not None and not preds.empty:
        ui.kv_table(
            [
                ("Prediction count", str(len(preds))),
                ("Forecast horizon", "1 step"),
                ("Last actual pH", f"{float(preds.iloc[-1]['actual']):.4f}"),
                ("Last predicted pH", f"{float(preds.iloc[-1]['predicted']):.4f}"),
            ]
        )
    ui.section("Performance")
    ui.metrics_row(metrics)
    ui.section("Visualization")
    if preds is not None and not preds.empty:
        ui.line_chart(preds, "Timestamp", ["actual", "predicted"], ["Actual", "Predicted"], "pH", f"Actual vs {title}")


def _render_hybrid_content(data: dict) -> None:
    comp = data["hybrid_components"]
    if comp is not None and not comp.empty:
        ui.section("Component Outputs")
        cols = [c for c in ["Timestamp", "actual", "arima_prediction", "lstm_residual_prediction", "hybrid_prediction"] if c in comp.columns]
        st.dataframe(comp[cols].tail(15), use_container_width=True, hide_index=True)
        ui.section("Visualization")
        y_cols = [c for c in ["actual", "arima_prediction", "hybrid_prediction"] if c in comp.columns]
        ui.line_chart(comp, "Timestamp", y_cols, ["Actual", "ARIMA", "Hybrid"], "pH", "Actual vs ARIMA vs Hybrid")
    else:
        preds = data["model_predictions"].get("arima_lstm_hybrid")
        if preds is not None and not preds.empty:
            ui.line_chart(preds, "Timestamp", ["actual", "predicted"], ["Actual", "Hybrid"], "pH", "Actual vs Hybrid prediction")
        else:
            st.caption("Component outputs not available from current run.")
    meta = data["hybrid_summary"]
    if meta:
        ui.section("Integration")
        st.write(meta.get("algorithm", HYBRID_DESC))
        if meta.get("arima_summary"):
            st.json(meta["arima_summary"])


def _render_hybrid(data: dict) -> None:
    ui.inline_stage_intro("ARIMA + LSTM Hybrid", HYBRID_DESC, model_status(data["summary"], data["failures"], "arima_lstm_hybrid"))
    _render_hybrid_content(data)


def _render_arlstima(data: dict) -> None:
    ui.inline_stage_intro("PROPOSED ARLSTMIMA FRAMEWORK", "Chapter 3 main novelty — ARIMA-LSTM hybrid forecasting.", model_status(data["summary"], data["failures"], "arima_lstm_hybrid"))
    st.markdown(
        f'<div style="border:2px solid #2563eb;padding:12px;background:#f0f6ff;border-radius:4px;">'
        f"<strong>Proposed Method:</strong> {HYBRID_DESC}</div>",
        unsafe_allow_html=True,
    )
    _render_hybrid_content(data)
    ui.section("Performance (ARLSTMIMA)")
    ui.metrics_row(model_metrics(data["comparison"], "arima_lstm_hybrid"))


def _render_ph_prediction(data: dict) -> None:
    ui.inline_stage_intro("pH Prediction", "User-facing forecast from the selected model.", "completed")
    metrics, preds = _main_result(data)
    if preds is None or preds.empty:
        st.caption("Not available from current run.")
        return
    last = preds.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current pH", f"{float(last['actual']):.3f}")
    c2.metric("Predicted pH", f"{float(last['predicted']):.3f}")
    c3.metric("Forecast Horizon", "1 step")
    c4.metric("Model", label((data["summary"] or {}).get("best_model", "—")))
    ui.section("Actual vs Predicted pH")
    ui.line_chart(preds, "Timestamp", ["actual", "predicted"], ["Actual", "Predicted"], "pH", "pH forecast (test set)")


def _render_comparison(data: dict) -> None:
    ui.inline_stage_intro("Model Comparison", "Compare all evaluated forecasting models on the test set.", "completed")
    comp = data["comparison"]
    if comp is None or comp.empty:
        st.caption("Not available from current run.")
        return
    show = comp.copy()
    show["Model"] = show["Model"].map(label)
    st.dataframe(show.sort_values("RMSE"), use_container_width=True, hide_index=True)
    best = best_model_row(comp)
    if best:
        st.caption(f"Best model by RMSE: **{label(best['Model'])}** (RMSE {best['RMSE']:.4f})")
    ui.section("Visualization")
    chart_df = comp.copy()
    chart_df["ModelLabel"] = chart_df["Model"].map(label)
    ui.bar_chart(chart_df, "ModelLabel", "RMSE", "RMSE by Model")
    ui.bar_chart(chart_df, "ModelLabel", "R2", "R² by Model", color=ui.C_OK)


def _render_final(data: dict) -> None:
    ui.inline_stage_intro("FINAL pH FORECASTING RESULT", "Summary of Chapter 3 outcomes (current run).", "completed")
    metrics, preds = _main_result(data)
    comp = data["comparison"]
    best = best_model_row(comp)
    if preds is not None and not preds.empty:
        last = preds.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Current pH", f"{float(last['actual']):.3f}")
        c2.metric("Predicted pH", f"{float(last['predicted']):.3f}")
        c3.metric("Best Model", label(best["Model"]) if best else "—")
    ui.section("Best Model Metrics")
    ui.metrics_row(metrics if metrics else (best and {k: best.get(k) for k in ["MAE", "MSE", "RMSE", "MAPE", "R2"]}))
    ui.section("Model Ranking")
    if data["ranking"] is not None:
        st.dataframe(data["ranking"], use_container_width=True, hide_index=True)
    ui.section("pH Status")
    if preds is not None and not preds.empty:
        ph = float(preds.iloc[-1]["actual"])
        if 6.5 <= ph <= 8.5:
            ui.status_pill("NORMAL — pH within acceptable range", "normal")
        else:
            ui.status_pill("OUT OF TYPICAL RANGE", "warning")
    if best:
        st.write(
            f"Based on test-set RMSE ({best['RMSE']:.4f}), **{label(best['Model'])}** "
            f"achieved the lowest error among evaluated models in this run."
        )
    with st.expander("Thesis Reference (not current run)"):
        st.caption("Values reported in the thesis manuscript only.")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Model": "ARIMA", "MAE": 0.284, "RMSE": 0.381, "MAPE": 6.82, "R2": 0.914},
                    {"Model": "LSTM", "MAE": 0.152, "RMSE": 0.263, "MAPE": 3.76, "R2": 0.973},
                    {"Model": "ARLSTMIMA", "MAE": 0.108, "RMSE": 0.202, "MAPE": 2.61, "R2": 0.989},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    ui.section("Next Stage")
    st.write("End of Chapter 3 workflow.")


def render_chapter3_page() -> None:
    st.markdown("### Development of an AIoT-Based Water Quality Monitoring System with the ARLSTMIMA Framework")
    st.caption("Chapter 3 — pH / Water Quality Forecasting")

    data = load_chapter3()
    if not data["summary"]:
        st.warning("Chapter 3 results not found. Run: python run_all.py --full --chapter3")
        return

    metrics, preds = _main_result(data)
    ui.section("Current Research Result (Current Run)")
    ui.status_badge("completed" if data["summary"].get("status") == "success" else "failed")
    if data["last_run"]:
        st.caption(f"Last run: {data['last_run']}")
    if preds is not None and not preds.empty:
        last = preds.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current pH", f"{float(last['actual']):.3f}")
        c2.metric("Predicted pH", f"{float(last['predicted']):.3f}")
        c3.metric("Horizon", "1 step")
        c4.metric("Best Model", label(data["summary"].get("best_model", "—")))
    else:
        st.caption("Main prediction not available from current run.")

    renderers = {
        "data": lambda: _render_data(data),
        "preprocessing": lambda: _render_preprocessing(data),
        "arima": lambda: _render_model_stage(data, "arima", "ARIMA Forecasting", "ARIMA models the time-series component of pH."),
        "lstm": lambda: _render_model_stage(data, "lstm", "LSTM Forecasting", "LSTM captures non-linear temporal patterns in pH."),
        "hybrid": lambda: _render_hybrid(data),
        "arlstima": lambda: _render_arlstima(data),
        "ph_prediction": lambda: _render_ph_prediction(data),
        "model_comparison": lambda: _render_comparison(data),
        "final_result": lambda: _render_final(data),
    }
    ui.render_inline_pipeline("ch3", _stages(data), renderers)

    with st.expander("Technical Details"):
        st.write(f"**Result file:** `{data['summary_path']}`")
        st.write(f"**Successful models:** {', '.join(label(m) for m in (data['summary'].get('successful_models') or []))}")
        fail = data["summary"].get("failed_models") or []
        if fail:
            st.write(f"**Failed models:** {', '.join(label(m) for m in fail)}")
