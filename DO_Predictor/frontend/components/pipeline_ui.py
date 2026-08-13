"""Shared pipeline UI — inline stage expansion, detail panels, charts."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

C_ACTUAL = "#374151"
C_PRED = "#2563eb"
C_OK = "#16a34a"
C_WARN = "#d97706"
C_ERR = "#dc2626"
C_MUTED = "#6b7280"
C_BORDER = "#d1d5db"
C_SELECTED = "#2563eb"

STATUS_ICON = {
    "completed": "✓ COMPLETED",
    "failed": "! FAILED",
    "running": "● RUNNING",
    "not_run": "○ NOT RUN",
    "unavailable": "○ NOT RUN",
}


def init_expanded(page_key: str, valid_stages: list[str]) -> str | None:
    qp = st.query_params.get("stage")
    key = f"{page_key}_expanded"
    if qp and qp in valid_stages:
        st.session_state[key] = qp
    return st.session_state.get(key)


def toggle_expanded(page_key: str, stage_id: str) -> None:
    key = f"{page_key}_expanded"
    if st.session_state.get(key) == stage_id:
        st.session_state[key] = None
        if "stage" in st.query_params:
            del st.query_params["stage"]
    else:
        st.session_state[key] = stage_id
        st.query_params["stage"] = stage_id


def section(title: str) -> None:
    st.markdown(f"**{title}**")
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {C_BORDER};margin:0.15rem 0 0.6rem;">',
        unsafe_allow_html=True,
    )


def inline_stage_intro(title: str, purpose: str, status: str) -> None:
    status_badge(status)
    st.markdown(f"**Purpose:** {purpose}")


def status_badge(status: str) -> None:
    colors = {
        "completed": C_OK,
        "failed": C_ERR,
        "running": C_WARN,
        "not_run": C_MUTED,
        "unavailable": C_MUTED,
    }
    label = STATUS_ICON.get(status, status.upper())
    color = colors.get(status, C_MUTED)
    st.markdown(f'<p style="color:{color};font-weight:600;margin:0 0 0.5rem 0;">{label}</p>', unsafe_allow_html=True)


def status_pill(text: str, level: str) -> None:
    colors = {"normal": C_OK, "warning": C_WARN, "alert": C_ERR, "na": C_MUTED}
    st.markdown(
        f'<p style="font-size:1.05rem;font-weight:600;color:{colors.get(level, C_MUTED)};">{text}</p>',
        unsafe_allow_html=True,
    )


def kv_table(rows: list[tuple[str, str]]) -> None:
    if not rows:
        st.caption("Not available from current run.")
        return
    st.dataframe(pd.DataFrame(rows, columns=["Field", "Value"]), use_container_width=True, hide_index=True)


def io_flow(input_text: str, process_text: str, output_text: str) -> None:
    st.markdown(
        f"""
        <div style="border:1px solid {C_BORDER};padding:12px;background:#fafafa;border-radius:4px;font-size:0.9rem;">
            <div><strong>INPUT</strong><br>{input_text}</div>
            <div style="text-align:center;margin:6px 0;color:{C_MUTED};">↓</div>
            <div><strong>PROCESS</strong><br>{process_text}</div>
            <div style="text-align:center;margin:6px 0;color:{C_MUTED};">↓</div>
            <div><strong>OUTPUT</strong><br>{output_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def line_chart(df: pd.DataFrame, x: str, y_cols: list[str], labels: list[str], y_label: str, title: str) -> None:
    if df is None or df.empty:
        st.caption("Not available from current run.")
        return
    fig = go.Figure()
    colors = [C_ACTUAL, C_PRED, C_WARN, C_ERR]
    for i, col in enumerate(y_cols):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[col],
                    name=labels[i] if i < len(labels) else col,
                    line=dict(color=colors[i % len(colors)], width=1.5),
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_label,
        template="plotly_white",
        height=360,
        margin=dict(l=48, r=16, t=44, b=44),
        legend=dict(orientation="h", y=1.08),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = C_PRED) -> None:
    if df is None or df.empty:
        st.caption("Not available from current run.")
        return
    fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=[color])
    fig.update_layout(template="plotly_white", height=320, paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


def metrics_row(metrics: dict | None, *, prefix: str = "Current Run") -> None:
    if not metrics:
        st.caption("Not available from current run.")
        return
    st.caption(prefix)
    cols = st.columns(5)
    cols[0].metric("MAE", f"{metrics.get('MAE', 0):.4f}" if metrics.get("MAE") is not None else "—")
    cols[1].metric("MSE", f"{metrics.get('MSE', 0):.4f}" if metrics.get("MSE") is not None else "—")
    cols[2].metric("RMSE", f"{metrics.get('RMSE', 0):.4f}" if metrics.get("RMSE") is not None else "—")
    cols[3].metric("MAPE (%)", f"{metrics.get('MAPE', 0):.2f}" if metrics.get("MAPE") is not None else "—")
    cols[4].metric("R²", f"{metrics.get('R2', 0):.4f}" if metrics.get("R2") is not None else "—")
    if metrics.get("n") is not None:
        st.caption(f"Evaluated observations: {metrics['n']}")


def render_inline_pipeline(
    page_key: str,
    stages: list[tuple[str, str, str]],
    renderers: dict,
) -> None:
    """Render pipeline with detail panels expanding inline at the clicked stage."""
    stage_ids = [s[0] for s in stages]
    expanded = init_expanded(page_key, stage_ids)

    st.markdown("##### RESEARCH PROCESS")
    if expanded and expanded in stage_ids:
        idx = stage_ids.index(expanded)
        nav1, nav2 = st.columns(2)
        with nav1:
            if idx > 0 and st.button("← Previous Stage", key=f"{page_key}_prev"):
                st.session_state[f"{page_key}_expanded"] = stage_ids[idx - 1]
                st.query_params["stage"] = stage_ids[idx - 1]
                st.rerun()
        with nav2:
            if idx < len(stage_ids) - 1 and st.button("Next Stage →", key=f"{page_key}_next"):
                st.session_state[f"{page_key}_expanded"] = stage_ids[idx + 1]
                st.query_params["stage"] = stage_ids[idx + 1]
                st.rerun()

    for i, (sid, label, status) in enumerate(stages):
        is_open = expanded == sid
        border = f"2px solid {C_SELECTED}" if is_open else f"1px solid {C_BORDER}"
        bg = "#f0f6ff" if is_open else "#ffffff"
        icon = STATUS_ICON.get(status, "")

        hdr, btn = st.columns([5, 1])
        with hdr:
            st.markdown(
                f'<div style="border:{border};background:{bg};border-radius:4px;padding:10px 14px;">'
                f'<span style="font-weight:600;">{label}</span> '
                f'<span style="color:{C_MUTED};font-size:0.82rem;">{icon}</span></div>',
                unsafe_allow_html=True,
            )
        with btn:
            if st.button("Collapse" if is_open else "Expand", key=f"{page_key}_tog_{sid}", use_container_width=True):
                toggle_expanded(page_key, sid)
                st.rerun()

        if is_open and sid in renderers:
            with st.container(border=True):
                renderers[sid]()

        if i < len(stages) - 1:
            st.markdown(
                f'<div style="text-align:center;color:{C_MUTED};font-size:1.1rem;margin:2px 0;">↓</div>',
                unsafe_allow_html=True,
            )
