"""Preprocessing utilities: scaling and split helpers (train-only fitting)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.research_config import MIN_FEATURE_COVERAGE, TEST_FRACTION, TIMESTAMP_COL, TRAIN_FRACTION, VALIDATION_FRACTION


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRACTION,
    val_frac: float = VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)
    return train, val, test


def split_period_strings(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict[str, str | None]:
    def _range(part: pd.DataFrame) -> str | None:
        if part.empty or TIMESTAMP_COL not in part.columns:
            return None
        return f"{part[TIMESTAMP_COL].min()} -> {part[TIMESTAMP_COL].max()}"

    return {"training_period": _range(train), "validation_period": _range(val), "test_period": _range(test)}


class TrainOnlyScaler:
    """StandardScaler fitted on training data only."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.columns_: list[str] | None = None

    def fit(self, train_df: pd.DataFrame, columns: list[str]) -> "TrainOnlyScaler":
        self.columns_ = columns
        self.scaler.fit(train_df[columns].astype(float))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("Scaler not fitted.")
        out = df.copy()
        out[self.columns_] = self.scaler.transform(out[self.columns_].astype(float))
        return out

    def fit_transform_train(self, train_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        self.fit(train_df, columns)
        return self.transform(train_df)


def drop_na_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    target_label = f"{target_col}_target"
    cols = [target_label] if target_label in df.columns else [target_col]
    return df.dropna(subset=cols).reset_index(drop=True)


def filter_features_by_coverage(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_coverage: float = 0.85,
) -> list[str]:
    """Drop engineered features with excessive missingness before row-wise dropna."""
    return [c for c in feature_cols if c in df.columns and df[c].notna().mean() >= min_coverage]


def drop_incomplete_feature_rows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Drop rows with NaN in any model feature or target before train/val/test split."""
    target_label = f"{target_col}_target"
    subset: list[str] = []
    for col in feature_cols:
        if col in df.columns:
            subset.append(col)
    if target_col in df.columns:
        subset.append(target_col)
    if target_label in df.columns:
        subset.append(target_label)
    if not subset:
        return df.reset_index(drop=True)
    return df.dropna(subset=subset).reset_index(drop=True)


def subsample_for_fast_mode(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.iloc[-max_rows:].reset_index(drop=True)


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_label: str,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    exclude = set(exclude_cols or [])
    keep = [c for c in feature_cols if c not in exclude and c in df.columns]
    meta_cols = [c for c in [TIMESTAMP_COL] if c in df.columns]
    X = df[meta_cols + keep]
    y = df[target_label]
    return X, y
