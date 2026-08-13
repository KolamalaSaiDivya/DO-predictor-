"""Pydantic request/response models for the FastAPI app."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QualityReportResponse(BaseModel):
    n_rows: int
    n_columns: int
    duplicate_row_count: int
    duplicate_timestamp_count: int
    record_number_is_canonical_order: bool
    timestamp_monotonic_raw: bool
    modal_cadence_minutes: float | None
    n_gaps_over_1h: int
    quality_columns_kept: list[str]
    quality_columns_dropped: list[str]
    notes: list[str]


class UploadResponse(BaseModel):
    filename: str
    n_rows: int
    n_columns: int
    quality_report: QualityReportResponse


class TrainRequest(BaseModel):
    model_name: str | None = Field(None, description="Registry key; omit to train every model in the registry")
    stage: str = Field("full_features", description="cleaned | smoothed | lag_features | full_features")


class TrainedModelMetrics(BaseModel):
    target: str
    model: str
    status: str
    train_time_seconds: float | None = None
    n_train: int | None = None
    n_test: int | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    test_mape: float | None = None
    test_r2: float | None = None


class TrainResponse(BaseModel):
    results: list[TrainedModelMetrics]


class PredictRequest(BaseModel):
    model_name: str | None = Field(None, description="Omit to use the best model on test MAE")


class PredictionPoint(BaseModel):
    timestamp: str
    actual: float | None
    predicted: float


class PredictResponse(BaseModel):
    target: str
    model_used: str
    low_value_threshold: float
    low_value_flag: bool
    latest_prediction: float
    recent: list[PredictionPoint]


class HistoryEntry(BaseModel):
    target: str
    model: str
    status: str
    test_mae: float | None = None
    test_rmse: float | None = None
    timestamp: str


class HistoryResponse(BaseModel):
    entries: list[HistoryEntry]


class MetricsResponse(BaseModel):
    target: str
    rows: list[dict[str, Any]]
