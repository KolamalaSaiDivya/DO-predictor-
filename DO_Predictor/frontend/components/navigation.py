"""Top-level page navigation — session state + query params, native Streamlit buttons."""

from __future__ import annotations

import streamlit as st

PAGE_CH3 = "chapter3"
PAGE_CH5 = "chapter5"

CH3_SHORT = "AIoT Water Quality Monitoring & ARLSTMIMA Forecasting"
CH5_SHORT = "DO Monitoring & Forecasting"

CH3_FULL = (
    "Development of an AIoT-Based Water Quality Monitoring System "
    "with the ARLSTMIMA Framework"
)
CH5_FULL = (
    "Application of the Forecasting Framework to Dissolved Oxygen "
    "Monitoring and Forecasting in Shrimp Aquaculture"
)


def _init_page_state() -> None:
    qp = st.query_params.get("page")
    if qp in (PAGE_CH3, PAGE_CH5):
        st.session_state["current_page"] = qp
    elif "current_page" not in st.session_state:
        st.session_state["current_page"] = PAGE_CH3
    # Preserve stage param across reruns (handled in pipeline_ui.init_stage_state)


def _set_page(page: str) -> None:
    st.session_state["current_page"] = page
    st.query_params["page"] = page
    if page == PAGE_CH3:
        st.session_state.pop("ch5_expanded", None)
    else:
        st.session_state.pop("ch3_expanded", None)


def render_page_selector() -> str:
    """Render selector; return active page key."""
    _init_page_state()
    current = st.session_state["current_page"]

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none; }
        .block-container { padding-top: 1rem; max-width: 1080px; }
        [data-testid="stMetricValue"] { font-size: 1.25rem; }
        div[data-testid="column"] .stButton > button {
            min-height: 3.25rem;
            font-weight: 600;
            white-space: normal;
            text-align: center;
            line-height: 1.3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## RESEARCH FORECASTING SYSTEM")

    _, refresh_col = st.columns([5, 1])
    with refresh_col:
        if st.button("Refresh Results", type="primary", use_container_width=True, key="refresh_results"):
            from frontend.components.research_results import clear_result_caches
            clear_result_caches()
            st.cache_data.clear()
            _init_page_state()
            st.toast("Results cache cleared — reloading latest backend files.")
            st.rerun()

    col3, col5 = st.columns(2, gap="medium")

    with col3:
        ch3_selected = current == PAGE_CH3
        border = "2px solid #2563eb" if ch3_selected else "2px solid #d1d5db"
        bg = "#f0f6ff" if ch3_selected else "#ffffff"
        st.markdown(
            f"""
            <div style="border:{border};background:{bg};border-radius:4px;padding:12px 14px;margin-bottom:8px;">
                <div style="font-size:0.8rem;font-weight:600;color:#2563eb;margin-bottom:6px;">CHAPTER 3</div>
                <div style="font-size:0.78rem;color:#6b7280;line-height:1.4;">{CH3_FULL}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            CH3_SHORT,
            key="btn_chapter3",
            use_container_width=True,
            type="primary" if ch3_selected else "secondary",
        ):
            if not ch3_selected:
                _set_page(PAGE_CH3)
                st.rerun()

    with col5:
        ch5_selected = current == PAGE_CH5
        border = "2px solid #2563eb" if ch5_selected else "2px solid #d1d5db"
        bg = "#f0f6ff" if ch5_selected else "#ffffff"
        st.markdown(
            f"""
            <div style="border:{border};background:{bg};border-radius:4px;padding:12px 14px;margin-bottom:8px;">
                <div style="font-size:0.8rem;font-weight:600;color:#2563eb;margin-bottom:6px;">CHAPTER 5</div>
                <div style="font-size:0.78rem;color:#6b7280;line-height:1.4;">{CH5_FULL}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            CH5_SHORT,
            key="btn_chapter5",
            use_container_width=True,
            type="primary" if ch5_selected else "secondary",
        ):
            if not ch5_selected:
                _set_page(PAGE_CH5)
                st.rerun()

    if ch3_selected:
        st.caption("Selected: Chapter 3 — pH / Water Quality Forecasting")
    else:
        st.caption("Selected: Chapter 5 — Dissolved Oxygen Forecasting & Decision Support")

    st.divider()
    return st.session_state["current_page"]
