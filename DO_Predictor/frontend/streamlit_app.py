"""Thesis research result viewer — Chapter 3 and Chapter 5."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from frontend.components.navigation import PAGE_CH3, render_page_selector
from frontend.components.research_dashboard import render_chapter3_page, render_chapter5_page

st.set_page_config(page_title="Research Forecasting System", layout="wide", initial_sidebar_state="collapsed")

current_page = render_page_selector()

if current_page == PAGE_CH3:
    render_chapter3_page()
else:
    render_chapter5_page()
