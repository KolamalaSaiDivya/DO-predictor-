"""Chapter 5 interactive process page — real backend results only."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from config.research_config import LOW_DO_THRESHOLD, WARNING_DO_THRESHOLD
from frontend.components import pipeline_ui as ui
from frontend.components.research_results import (
    best_model_row,
    feature_type_counts,
    fs_label,
    label,
    load_chapter5,
    model_metrics,
    prep_target_stats,
)

CH5_STAGES = [
    "data",
    "preprocessing",
    "feature_engineering",
    "rfe",
    "csf",
    "feature_set_comparison",
    "model_comparison",
    "best_model",
    "do_prediction",
    "low_do",
    "final_result",
]


def _stages(data: dict) -> list[tuple[str, str, str]]:
    s = data["summary"]
    has_fe = data["feature_dictionary"] is not None
    has_rfe = data["rfe_features"] is not None
    has_cfs = data["cfs_features"] is not None
    has_comp = data["comparison"] is not None and not data["comparison"].empty
    return [
        ("data", "DO DATA", "completed" if data["load_report"] else "failed"),
        ("preprocessing", "PREPROCESSING", "completed" if data["preprocessing"] is not None else "not_run"),
        ("feature_engineering", "FEATURE ENGINEERING", "completed" if has_fe else "not_run"),
        ("rfe", "RFE", "completed" if has_rfe else "not_run"),
        ("csf", "CFS", "completed" if has_cfs else "not_run"),
        ("feature_set_comparison", "FEATURE-SET COMPARISON", "completed" if has_comp else "not_run"),
        ("model_comparison", "MODEL COMPARISON", "completed" if has_comp else "not_run"),
        ("best_model", "BEST FEATURE SET + MODEL", "completed" if has_comp else "not_run"),
        ("do_prediction", "DO PREDICTION", "completed" if data["predictions"] is not None and not data["predictions"].empty else "not_run"),
        ("low_do", "LOW-DO DECISION SUPPORT", "completed" if LOW_DO_THRESHOLD is not None else "failed"),
        ("final_result", "FINAL RESULT", "completed" if s and s.get("status") == "success" else "failed"),
    ]


def _best_fs_row(data: dict) -> dict | None:
    comp = data["comparison"]
    if comp is None or comp.empty:
        return None
    if "Feature_Set" not in comp.columns:
        return best_model_row(comp)
    best_per_fs = []
    for fs in comp["Feature_Set"].unique():
        sub = comp[comp["Feature_Set"] == fs].sort_values("RMSE")
        if not sub.empty:
            best_per_fs.append(sub.iloc[0].to_dict())
    if not best_per_fs:
        return None
    return min(best_per_fs, key=lambda r: r["RMSE"])


def _render_data(data: dict) -> None:
    ui.inline_stage_intro("Dissolved Oxygen Data", "Load and validate the Chapter 5 DO dataset.", "completed" if data["load_report"] else "failed")
    lr = data["load_report"] or {}
    ui.kv_table(
        [
            ("Dataset file", lr.get("filename", "chapter5_do_dataset.csv")),
            ("Source path", data["data_path"]),
            ("Total rows", str(lr.get("n_rows", "—"))),
            ("Total columns", str(lr.get("n_columns", "—"))),
            ("Target column", "DO"),
            ("Unit", "mg/L"),
            ("Timestamp column", "Timestamp"),
            ("Time range", " → ".join(lr.get("time_range", [])) if lr.get("time_range") else "—"),
        ]
    )
    ui.section("Data Sample")
    if data["cleaned_sample"] is not None and not data["cleaned_sample"].empty:
        st.dataframe(data["cleaned_sample"].head(10), use_container_width=True, hide_index=True)
    ui.section("Target Statistics (DO)")
    stats = prep_target_stats(data["preprocessing"], "DO")
    if stats:
        ui.kv_table([(k, str(v)) for k, v in stats.items()])
    ui.section("Visualization")
    sample = data["cleaned_sample"]
    if sample is not None and "DO" in sample.columns:
        x = "Timestamp" if "Timestamp" in sample.columns else sample.columns[0]
        ui.line_chart(sample, x, ["DO"], ["DO (mg/L)"], "DO (mg/L)", "DO time series (sample)")


def _render_preprocessing(data: dict) -> None:
    ui.inline_stage_intro("Data Preprocessing", "Clean, order, and prepare DO data for forecasting.", "completed")
    s = data["summary"] or {}
    h = data["horizon"] or {}
    ui.io_flow(
        f"{s.get('dataset_rows', '—')} observations",
        "Missing handling · Chronological ordering · Target shift · Train/val/test split",
        f"train {s.get('train_size')} · val {s.get('validation_size')} · test {s.get('test_size')} · horizon {h.get('target_horizon_minutes', '—')} min",
    )
    if data["preprocessing"] is not None:
        st.dataframe(data["preprocessing"], use_container_width=True, hide_index=True)


def _render_feature_engineering(data: dict) -> None:
    ui.inline_stage_intro("Feature Engineering", "Create lag, rolling, and temporal features from sensor columns.", "completed" if data["feature_dictionary"] is not None else "not_run")
    ft = feature_type_counts(data["feature_dictionary"])
    s = data["summary"] or {}
    n_eng = ft.get("total") if ft else None
    if n_eng is None and data["feature_dictionary"] is not None:
        n_eng = len(data["feature_dictionary"])
    ui.kv_table(
        [
            ("Candidate features", str(s.get("candidate_feature_count", "—"))),
            ("Engineered features (dictionary)", str(n_eng if n_eng is not None else "—")),
            ("Lag features", str(ft.get("lag", "—"))),
            ("Rolling features", str(ft.get("rolling", "—"))),
            ("RFE selected", str(s.get("rfe_feature_count", "—"))),
            ("CFS selected", str(s.get("cfs_feature_count", "—"))),
            ("RFE+CFS final", str(s.get("rfe_cfs_feature_count", "—"))),
        ]
    )
    fdict = data["feature_dictionary"]
    if fdict is not None and not fdict.empty:
        ui.section("Feature Dictionary (sample)")
        st.dataframe(fdict.head(20), use_container_width=True, hide_index=True)
        with st.expander("View All Features"):
            st.dataframe(fdict, use_container_width=True, hide_index=True)
    ui.section("Validation")
    st.write("Lag/rolling features use causal trailing windows only (no future leakage in feature construction).")


def _render_rfe(data: dict) -> None:
    ui.inline_stage_intro("RFE Feature Selection", "Recursive Feature Elimination on training data.", "completed" if data["rfe_features"] is not None else "not_run")
    s = data["summary"] or {}
    ui.kv_table(
        [
            ("Input feature count", str(s.get("candidate_feature_count", "—"))),
            ("Selected feature count", str(s.get("rfe_feature_count", "—"))),
            ("Estimator", "RandomForestRegressor (sklearn RFE)"),
            ("Selection data", "Training set only"),
        ]
    )
    if data["rfe_rankings"] is not None:
        ui.section("Feature Rankings")
        show = data["rfe_rankings"].copy()
        if "selected" in show.columns:
            show = show.sort_values(["selected", "ranking"], ascending=[False, True])
        st.dataframe(show.head(30), use_container_width=True, hide_index=True)
    if data["rfe_val_scores"] is not None:
        ui.section("Validation RMSE by Feature Count")
        st.dataframe(data["rfe_val_scores"], use_container_width=True, hide_index=True)


def _render_csf(data: dict) -> None:
    ui.inline_stage_intro("CFS Feature Selection", "Correlation-based Feature Selection on RFE-reduced pool (CSF in thesis).", "completed" if data["cfs_features"] is not None else "not_run")
    s = data["summary"] or {}
    ui.kv_table(
        [
            ("Method", "CFS merit function (Correlation-based Feature Selection)"),
            ("Input pool", "RFE-selected features"),
            ("Selected count", str(s.get("cfs_feature_count", "—"))),
            ("Final RFE+CFS count", str(s.get("rfe_cfs_feature_count", "—"))),
            ("Selection data", "Training set only"),
        ]
    )
    if data["cfs_scores"] is not None:
        ui.section("CFS Scores")
        st.dataframe(data["cfs_scores"].head(30), use_container_width=True, hide_index=True)
    if data["cfs_features"] is not None:
        ui.section("Selected Features")
        st.dataframe(data["cfs_features"], use_container_width=True, hide_index=True)


def _render_feature_set_comparison(data: dict) -> None:
    ui.inline_stage_intro("Feature-Set Comparison", "Compare Full vs RFE vs CFS vs RFE+CFS strategies.", "completed")
    comp = data["comparison"]
    s = data["summary"] or {}
    evaluated = s.get("evaluated_feature_sets") or []
    rows = []
    for fs_key in ["all_features", "rfe", "cfs", "rfe_cfs"]:
        sub = comp[comp["Feature_Set"] == fs_key] if comp is not None and not comp.empty else pd.DataFrame()
        if sub.empty:
            rows.append({"Feature Set": fs_label(fs_key), "Status": "Not evaluated in current run"})
            continue
        best = sub.sort_values("RMSE").iloc[0]
        rows.append(
            {
                "Feature Set": fs_label(fs_key),
                "Features": int(best.get("Number_of_Features", 0)),
                "Best Model": label(best["Model"]),
                "MAE": f"{best['MAE']:.4f}",
                "RMSE": f"{best['RMSE']:.4f}",
                "MAPE": f"{best['MAPE']:.2f}",
                "R²": f"{best['R2']:.4f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Evaluated in current run: {', '.join(fs_label(f) for f in evaluated)}")
    best_fs = _best_fs_row(data)
    if best_fs:
        st.write(f"**Best feature-selection strategy (by RMSE):** {fs_label(best_fs['Feature_Set'])} — {label(best_fs['Model'])} (RMSE {best_fs['RMSE']:.4f})")


def _render_model_comparison(data: dict) -> None:
    ui.inline_stage_intro("Model Comparison", "All models × feature sets evaluated on held-out test data.", "completed")
    comp = data["comparison"]
    if comp is None or comp.empty:
        st.caption("Not available from current run.")
        return
    show = comp.copy()
    show["Model"] = show["Model"].map(label)
    show["Feature_Set"] = show["Feature_Set"].map(fs_label)
    st.dataframe(show.sort_values("RMSE"), use_container_width=True, hide_index=True)
    chart = comp.groupby("Feature_Set", as_index=False)["RMSE"].min()
    chart["Feature_Set"] = chart["Feature_Set"].map(fs_label)
    ui.bar_chart(chart, "Feature_Set", "RMSE", "Best RMSE by Feature Set")


def _render_best_model(data: dict) -> None:
    ui.inline_stage_intro("Best Feature Set + Model", "Selection from validation RMSE on RFE+CFS features.", "completed")
    s = data["summary"] or {}
    best_fs = _best_fs_row(data)
    comp = data["comparison"]
    ui.kv_table(
        [
            ("Best feature-selection strategy (test RMSE)", fs_label(best_fs["Feature_Set"]) if best_fs else "—"),
            ("Best model for that strategy", label(best_fs["Model"]) if best_fs else "—"),
            ("Final proposed model (validation selection)", label(str(s.get("final_proposed_model", "—")))),
            ("Final feature set (RFE+CFS pipeline)", str(s.get("rfe_cfs_feature_count", "—")) + " features"),
        ]
    )
    if comp is not None and not comp.empty and s.get("final_proposed_model"):
        final_rows = comp[
            (comp["Feature_Set"] == "rfe_cfs") & (comp["Model"] == s.get("final_proposed_model"))
        ]
        if not final_rows.empty:
            ui.section("Final Model Test Metrics")
            row = final_rows.iloc[0]
            ui.metrics_row(
                {"MAE": row["MAE"], "MSE": row["MSE"], "RMSE": row["RMSE"], "MAPE": row["MAPE"], "R2": row["R2"]}
            )
    if s.get("final_selected_features"):
        ui.section("Selected Features")
        st.dataframe(pd.DataFrame({"feature_name": s["final_selected_features"]}), use_container_width=True, hide_index=True)


def _render_do_prediction(data: dict) -> None:
    ui.inline_stage_intro("DO Prediction", "10-minute-ahead dissolved oxygen forecast.", "completed")
    preds = data["predictions"]
    s = data["summary"] or {}
    h = data["horizon"] or {}
    if preds is None or preds.empty:
        st.caption("Not available from current run.")
        return
    last = preds.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current DO (mg/L)", f"{float(last['actual_DO']):.3f}")
    c2.metric("Predicted DO (mg/L)", f"{float(last['predicted_DO']):.3f}")
    c3.metric("Horizon", f"{h.get('target_horizon_minutes', last.get('forecast_horizon_minutes', '—'))} min")
    c4.metric("Model", label(str(s.get("final_proposed_model", last.get("model", "—")))))
    ui.line_chart(preds, "timestamp_input" if "timestamp_input" in preds.columns else preds.columns[0], ["actual_DO", "predicted_DO"], ["Actual", "Predicted"], "DO (mg/L)", "Actual vs Predicted DO")
    ui.metrics_row(s.get("final_metrics"))


def _render_low_do(data: dict) -> None:
    ui.inline_stage_intro("Low-DO Decision Support", "Threshold-based aquaculture decision support.", "completed" if LOW_DO_THRESHOLD is not None else "failed")
    if LOW_DO_THRESHOLD is None:
        st.error("LOW_DO_THRESHOLD is not configured.")
        return
    preds = data["predictions"]
    alerts = data["alerts"]
    st.write(f"**LOW_DO_THRESHOLD:** {LOW_DO_THRESHOLD} mg/L")
    if preds is not None and not preds.empty:
        last = preds.iloc[-1]
        pred_do = float(last["predicted_DO"])
        diff = pred_do - LOW_DO_THRESHOLD
        if alerts is not None and not alerts.empty:
            latest = alerts.iloc[-1]
            status = str(latest.get("status", "Normal"))
            level = "alert" if "Low DO" in status else ("warning" if status == "Warning" else "normal")
            ui.status_pill(status.upper(), level)
        elif pred_do < LOW_DO_THRESHOLD:
            ui.status_pill("LOW DO ALERT", "alert")
        else:
            ui.status_pill("NORMAL", "normal")
        st.write(f"Predicted DO: {pred_do:.3f} mg/L | Threshold: {LOW_DO_THRESHOLD} mg/L | Difference: {diff:+.3f} mg/L")
        if alerts is not None and not alerts.empty:
            st.dataframe(alerts.tail(15), use_container_width=True, hide_index=True)


def _render_final(data: dict) -> None:
    ui.inline_stage_intro("FINAL DO FORECASTING RESULT", "Chapter 5 summary (current run).", "completed")
    s = data["summary"] or {}
    preds = data["predictions"]
    fm = s.get("final_metrics") or {}
    if preds is not None and not preds.empty:
        last = preds.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current DO", f"{float(last['actual_DO']):.3f} mg/L")
        c2.metric("Predicted DO", f"{float(last['predicted_DO']):.3f} mg/L")
        c3.metric("Best Model", label(str(s.get("final_proposed_model", "—"))))
        c4.metric("Feature Set", fs_label(str(s.get("final_feature_set", "rfe_cfs"))))
    ui.section("Best Model Metrics")
    ui.metrics_row(fm)
    best_fs = _best_fs_row(data)
    if best_fs:
        st.write(
            f"Lowest test RMSE in this run: **{fs_label(best_fs['Feature_Set'])}** with "
            f"**{label(best_fs['Model'])}** (RMSE {best_fs['RMSE']:.4f}). "
            f"Final proposed model selected by validation RMSE on RFE+CFS features."
        )
    with st.expander("Thesis Reference (not current run)"):
        st.caption("Values reported in the thesis manuscript only — not overwritten by current run.")
        st.write("Thesis-reported best model and metrics are available in the manuscript tables.")


def render_chapter5_page() -> None:
    st.markdown(
        "### Application of the Forecasting Framework to Dissolved Oxygen "
        "Monitoring and Forecasting in Shrimp Aquaculture"
    )
    st.caption("Chapter 5 — DO Forecasting & Decision Support")

    data = load_chapter5()
    if not data["summary"]:
        st.warning("Chapter 5 results not found. Run: python run_all.py --full --chapter5")
        return

    s = data["summary"]
    preds = data["predictions"]
    ui.section("Current Research Result (Current Run)")
    ui.status_badge("completed" if s.get("status") == "success" else "failed")
    if data["last_run"]:
        st.caption(f"Last run: {data['last_run']}")
    if preds is not None and not preds.empty:
        last = preds.iloc[-1]
        h = data["horizon"] or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current DO (mg/L)", f"{float(last['actual_DO']):.3f}")
        c2.metric("Predicted DO (mg/L)", f"{float(last['predicted_DO']):.3f}")
        c3.metric("Horizon", f"{h.get('target_horizon_minutes', '—')} min")
        c4.metric("Final Model", label(str(s.get("final_proposed_model", "—"))))

    renderers = {
        "data": lambda: _render_data(data),
        "preprocessing": lambda: _render_preprocessing(data),
        "feature_engineering": lambda: _render_feature_engineering(data),
        "rfe": lambda: _render_rfe(data),
        "csf": lambda: _render_csf(data),
        "feature_set_comparison": lambda: _render_feature_set_comparison(data),
        "model_comparison": lambda: _render_model_comparison(data),
        "best_model": lambda: _render_best_model(data),
        "do_prediction": lambda: _render_do_prediction(data),
        "low_do": lambda: _render_low_do(data),
        "final_result": lambda: _render_final(data),
    }
    ui.render_inline_pipeline("ch5", _stages(data), renderers)

    with st.expander("Technical Details"):
        st.write(f"**Result file:** `{data['summary_path']}`")
        st.write(f"**Selected features:** {', '.join(s.get('final_selected_features') or [])}")
