"""Publication-quality figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIG_DPI = 300


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_time_series(df: pd.DataFrame, timestamp_col: str, value_col: str, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(df[timestamp_col]), df[value_col], linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(value_col)
    _save(fig, output)


def plot_train_val_test_split(df: pd.DataFrame, timestamp_col: str, target_col: str, splits: tuple[int, int], output: Path) -> None:
    train_end, val_end = splits
    fig, ax = plt.subplots(figsize=(12, 4))
    ts = pd.to_datetime(df[timestamp_col])
    ax.plot(ts, df[target_col], color="gray", linewidth=0.6, label="Series")
    ax.axvspan(ts.iloc[0], ts.iloc[train_end - 1], alpha=0.15, color="green", label="Train")
    ax.axvspan(ts.iloc[train_end], ts.iloc[val_end - 1], alpha=0.15, color="orange", label="Validation")
    ax.axvspan(ts.iloc[val_end], ts.iloc[-1], alpha=0.15, color="red", label="Test")
    ax.set_title("Chronological Train / Validation / Test Split")
    ax.legend()
    _save(fig, output)


def plot_actual_vs_predicted(timestamps, y_true, y_pred, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timestamps, y_true, label="Actual", linewidth=0.9)
    ax.plot(timestamps, y_pred, label="Predicted", linewidth=0.9, alpha=0.8)
    ax.set_title(title)
    ax.legend()
    _save(fig, output)


def plot_metric_comparison(comparison_df: pd.DataFrame, metric: str, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=comparison_df, x="Model", y=metric, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    _save(fig, output)


def plot_loss_history(history_df: pd.DataFrame, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if "loss" in history_df.columns:
        ax.plot(history_df["loss"], label="Training loss")
    if "val_loss" in history_df.columns:
        ax.plot(history_df["val_loss"], label="Validation loss")
    ax.set_title(title)
    ax.legend()
    _save(fig, output)


def plot_residuals(y_true, y_pred, title: str, output: Path) -> None:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title(title)
    _save(fig, output)


def plot_correlation_heatmap(corr: pd.DataFrame, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    _save(fig, output)


def plot_feature_count_vs_error(scores: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(scores["feature_count"], scores["rmse"], marker="o")
    ax.set_xlabel("Feature count")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Feature Count vs Validation Error")
    _save(fig, output)


def plot_feature_merit(scores: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=scores, x="feature_name", y="merit", ax=ax)
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("CFS Feature Merit")
    _save(fig, output)
