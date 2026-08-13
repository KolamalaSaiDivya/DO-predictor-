"""Shape / NaN / leakage sanity checks for the data pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PRIMARY_TARGET, RESAMPLE_FREQ, TIMESTAMP_COL
from src.pipeline import (
    build_base_artifacts,
    build_stage_dataframe,
    chronological_split,
    longest_contiguous_block,
    prepare_model_data,
)


@pytest.fixture(scope="module")
def base():
    return build_base_artifacts()


def test_raw_data_shape(base):
    assert base.raw.shape[0] > 30000
    assert "Dissolved Oxygen" in base.raw.columns


def test_ph_is_excluded_everywhere(base):
    """This is a Dissolved-Oxygen-only system - pH must not survive loading."""
    assert "pH" not in base.raw.columns
    assert "pH [quality]" not in base.raw.columns
    assert "pH" not in base.cleaned.columns
    assert "pH" not in base.smoothed.columns


def test_cleaned_data_has_no_duplicate_timestamps(base):
    assert base.cleaned[TIMESTAMP_COL].duplicated().sum() == 0


def test_cleaned_data_is_uniformly_spaced(base):
    diffs = base.cleaned[TIMESTAMP_COL].diff().dropna()
    expected = pd.Timedelta(RESAMPLE_FREQ)
    assert (diffs == expected).all()


def test_quality_columns_were_detected_as_populated(base):
    assert len(base.quality_report.quality_columns_kept) > 0
    assert len(base.quality_report.quality_columns_dropped) == 0


def test_target_shift_has_no_leakage(base):
    """target_target at row i must equal the raw target one step later, not
    something derivable from features at row i itself."""
    stage_df = build_stage_dataframe(base, PRIMARY_TARGET, "cleaned")
    non_na = stage_df.dropna(subset=[f"{PRIMARY_TARGET}_target", PRIMARY_TARGET]).reset_index(drop=True)
    shifted = non_na[PRIMARY_TARGET].shift(-1)
    matches = np.isclose(non_na[f"{PRIMARY_TARGET}_target"], shifted, equal_nan=True)
    # last row's shift(-1) is NaN by construction, exclude it
    assert matches[:-1].mean() > 0.99


def test_full_features_stage_has_no_nan_after_dropna(base):
    stage_df = build_stage_dataframe(base, PRIMARY_TARGET, "full_features")
    clean = stage_df.dropna()
    assert clean.isna().sum().sum() == 0
    assert len(clean) > 1000


def test_chronological_split_is_time_ordered_and_non_overlapping(base):
    stage_df = build_stage_dataframe(base, PRIMARY_TARGET, "full_features").dropna().reset_index(drop=True)
    train, val, test = chronological_split(stage_df)
    assert train[TIMESTAMP_COL].max() <= val[TIMESTAMP_COL].min()
    assert val[TIMESTAMP_COL].max() <= test[TIMESTAMP_COL].min()
    assert len(train) + len(val) + len(test) == len(stage_df)


def test_longest_contiguous_block_is_actually_contiguous(base):
    stage_df = build_stage_dataframe(base, PRIMARY_TARGET, "full_features").dropna().reset_index(drop=True)
    block = longest_contiguous_block(stage_df)
    diffs = block[TIMESTAMP_COL].diff().dropna()
    assert (diffs == pd.Timedelta(RESAMPLE_FREQ)).all()
    assert len(block) < len(stage_df)


def test_prepare_model_data_train_precedes_test(base):
    stage_df = build_stage_dataframe(base, PRIMARY_TARGET, "full_features")
    splits = prepare_model_data(stage_df, PRIMARY_TARGET, "linear_regression")
    X_train, _ = splits["train"]
    X_test, _ = splits["test"]
    assert X_train[TIMESTAMP_COL].max() <= X_test[TIMESTAMP_COL].min()


def test_prepare_model_data_no_nan_in_features(base):
    stage_df = build_stage_dataframe(base, PRIMARY_TARGET, "full_features")
    splits = prepare_model_data(stage_df, PRIMARY_TARGET, "linear_regression")
    for X, y in splits.values():
        assert X.drop(columns=["Timestamp"]).isna().sum().sum() == 0
        assert y.isna().sum() == 0
