"""Tests for research data loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config.research_config import CHAPTER5_DATA_PATH
from src.research_data_loader import detect_timestamp_column, load_research_dataset, normalize_column_names


@pytest.fixture(scope="module")
def sample_df():
    if not CHAPTER5_DATA_PATH.exists():
        pytest.skip("Chapter 5 dataset not available")
    df, _ = load_research_dataset(CHAPTER5_DATA_PATH)
    return df


def test_chapter5_dataset_loads(sample_df):
    assert len(sample_df) > 1000
    assert "DO" in sample_df.columns


def test_timestamp_detection(sample_df):
    col = detect_timestamp_column(sample_df)
    assert col == "Timestamp"


def test_column_alias_mapping():
    df = pd.DataFrame({"dissolved oxygen": [1.0], "Timestamp": ["2024-01-01"]})
    out, mapping = normalize_column_names(df)
    assert "DO" in out.columns
