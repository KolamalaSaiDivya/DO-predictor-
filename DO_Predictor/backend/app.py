"""FastAPI service wrapping the pipeline: upload a CSV, train models, pull
metrics/predictions/history, download the thesis report bundle.

Single-process, in-memory state - this is a research/demo app for a thesis,
not a production multi-tenant service, so a module-level singleton is the
right amount of infrastructure.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.schemas import (
    HistoryEntry,
    HistoryResponse,
    MetricsResponse,
    PredictionPoint,
    PredictRequest,
    PredictResponse,
    QualityReportResponse,
    TrainedModelMetrics,
    TrainRequest,
    TrainResponse,
    UploadResponse,
)
from config.research_config import CHAPTER3_RESULTS, CHAPTER5_RESULTS, COMBINED_RESULTS, RESULTS_ROOT
from src.config import (
    LOW_DO_THRESHOLD_MGL,
    PRIMARY_TARGET,
    RAW_DATA_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    SENSOR_COLUMNS,
    TIMESTAMP_COL,
    TRAINED_MODELS_DIR,
    get_logger,
)
from src.data_loader import load_raw_data
from src.evaluation import StageMetricLogger
from src.model_io import load_model, model_exists, save_model
from src.models.registry import MODEL_REGISTRY
from src.pipeline import build_base_artifacts, build_stage_dataframe, prepare_model_data, train_and_evaluate
from src.validation import validate_raw_data

logger = get_logger(__name__)

app = FastAPI(title="DO_Predictor API", description="Dissolved Oxygen forecasting benchmark service")

UPLOAD_PATH = RAW_DATA_DIR / "uploaded_current.csv"
HISTORY_PATH = TRAINED_MODELS_DIR / "history.json"


class AppState:
    def __init__(self) -> None:
        self.current_csv_path: Path | None = None
        self.base_artifacts = None
        self.quality_report = None
        self.models: dict[tuple[str, str], object] = {}

    def ensure_base_artifacts(self):
        if self.base_artifacts is None:
            self.base_artifacts = build_base_artifacts(csv_path=self.current_csv_path)
            self.quality_report = self.base_artifacts.quality_report
        return self.base_artifacts

    def invalidate(self) -> None:
        self.base_artifacts = None
        self.quality_report = None
        self.models = {}


state = AppState()


def _load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text())
    return []


def _append_history(entries: list[dict]) -> None:
    history = _load_history()
    history.extend(entries)
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2, default=str))


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "DO_Predictor", "status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile) -> UploadResponse:
    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    missing = [c for c in [TIMESTAMP_COL, *SENSOR_COLUMNS] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Uploaded CSV missing required columns: {missing}")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_PATH.write_bytes(raw_bytes)
    state.current_csv_path = UPLOAD_PATH
    state.invalidate()

    raw_df = load_raw_data(csv_path=UPLOAD_PATH, download_if_missing=False)
    report = validate_raw_data(raw_df)

    return UploadResponse(
        filename=file.filename or "uploaded.csv",
        n_rows=raw_df.shape[0],
        n_columns=raw_df.shape[1],
        quality_report=QualityReportResponse(**{
            k: v for k, v in report.to_dict().items()
            if k in QualityReportResponse.model_fields
        }),
    )


@app.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    base = state.ensure_base_artifacts()
    stage_df = build_stage_dataframe(base, PRIMARY_TARGET, request.stage)

    model_names = [request.model_name] if request.model_name else list(MODEL_REGISTRY)
    for name in model_names:
        if name not in MODEL_REGISTRY:
            raise HTTPException(status_code=422, detail=f"Unknown model '{name}'. Known: {sorted(MODEL_REGISTRY)}")

    metric_logger = StageMetricLogger()
    results: list[TrainedModelMetrics] = []
    history_entries: list[dict] = []

    for name in model_names:
        try:
            result = train_and_evaluate(name, PRIMARY_TARGET, stage_df, request.stage, metric_logger)
        except Exception as exc:
            logger.exception("Training failed for %s", name)
            results.append(TrainedModelMetrics(target=PRIMARY_TARGET, model=name, status="failed"))
            continue

        if result is None:
            results.append(TrainedModelMetrics(target=PRIMARY_TARGET, model=name, status="insufficient_data"))
            continue

        state.models[(PRIMARY_TARGET, name)] = result["model"]
        save_model(result["model"], PRIMARY_TARGET)

        test_metrics = result["test_metrics"]
        results.append(TrainedModelMetrics(
            target=PRIMARY_TARGET, model=name, status="ok",
            n_train=result["n_train"], n_test=result["n_test"],
            test_mae=test_metrics["mae"], test_rmse=test_metrics["rmse"],
            test_mape=test_metrics["mape"], test_r2=test_metrics["r2"],
        ))
        history_entries.append({
            "target": PRIMARY_TARGET, "model": name, "status": "ok",
            "test_mae": test_metrics["mae"], "test_rmse": test_metrics["rmse"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    _append_history(history_entries)
    return TrainResponse(results=results)


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    comparison_path = RESULTS_DIR / "model_comparison.json"
    rows: list[dict] = []
    if comparison_path.exists():
        rows = json.loads(comparison_path.read_text())
    return MetricsResponse(target=PRIMARY_TARGET, rows=rows)


def _get_or_load_model(model_name: str):
    key = (PRIMARY_TARGET, model_name)
    if key in state.models:
        return state.models[key]
    if model_exists(model_name, PRIMARY_TARGET):
        model = load_model(model_name, PRIMARY_TARGET)
        state.models[key] = model
        return model
    return None


def _best_model_name() -> str:
    comparison_path = RESULTS_DIR / "model_comparison.json"
    if comparison_path.exists():
        rows = [r for r in json.loads(comparison_path.read_text()) if r.get("status") == "ok"]
        if rows:
            return min(rows, key=lambda r: r["test_mae"])["model"]
    return "persistence"


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model_name = request.model_name or _best_model_name()
    model = _get_or_load_model(model_name)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model '{model_name}' - call /train first.",
        )

    base = state.ensure_base_artifacts()
    stage_df = build_stage_dataframe(base, PRIMARY_TARGET, "full_features")
    splits = prepare_model_data(stage_df, PRIMARY_TARGET, model_name)
    X_test, y_test = splits["test"]

    preds = model.predict(X_test)
    n = len(preds)
    y_true = y_test.iloc[-n:].to_numpy()
    timestamps = X_test[TIMESTAMP_COL].iloc[-n:].tolist()

    recent_n = min(50, n)
    recent = [
        PredictionPoint(timestamp=str(timestamps[i]), actual=float(y_true[i]), predicted=float(preds[i]))
        for i in range(n - recent_n, n)
    ]

    latest_prediction = float(preds[-1])
    low_flag = latest_prediction < LOW_DO_THRESHOLD_MGL

    return PredictResponse(
        target=PRIMARY_TARGET,
        model_used=model_name,
        low_value_threshold=LOW_DO_THRESHOLD_MGL,
        low_value_flag=low_flag,
        latest_prediction=latest_prediction,
        recent=recent,
    )


@app.get("/history", response_model=HistoryResponse)
def history() -> HistoryResponse:
    entries = [HistoryEntry(**e) for e in _load_history()]
    return HistoryResponse(entries=entries)


@app.get("/quality_report", response_model=QualityReportResponse)
def quality_report() -> QualityReportResponse:
    base = state.ensure_base_artifacts()
    report = base.quality_report.to_dict()
    return QualityReportResponse(**{k: v for k, v in report.items() if k in QualityReportResponse.model_fields})


@app.get("/download_report")
def download_report() -> StreamingResponse:
    if not REPORTS_DIR.exists() or not any(REPORTS_DIR.iterdir()):
        raise HTTPException(status_code=404, detail="No reports generated yet.")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in REPORTS_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(REPORTS_DIR))
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=do_predictor_report.zip"},
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Results file not found: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_records(path: Path) -> list[dict]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Results file not found: {path.name}")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/research/status")
def research_status() -> dict:
    """Return availability of thesis experiment result files."""
    return {
        "results_root": str(RESULTS_ROOT),
        "chapter3_summary_exists": (CHAPTER3_RESULTS / "metrics" / "chapter3_summary.json").exists(),
        "chapter5_summary_exists": (CHAPTER5_RESULTS / "metrics" / "chapter5_summary.json").exists(),
        "combined_summary_exists": (COMBINED_RESULTS / "results_summary.json").exists(),
    }


@app.get("/research/chapter3/summary")
def chapter3_summary() -> dict:
    return _read_json(CHAPTER3_RESULTS / "metrics" / "chapter3_summary.json")


@app.get("/research/chapter3/comparison")
def chapter3_comparison() -> list[dict]:
    return _read_csv_records(CHAPTER3_RESULTS / "metrics" / "chapter3_model_comparison.csv")


@app.get("/research/chapter5/summary")
def chapter5_summary() -> dict:
    return _read_json(CHAPTER5_RESULTS / "metrics" / "chapter5_summary.json")


@app.get("/research/chapter5/comparison")
def chapter5_comparison() -> list[dict]:
    return _read_csv_records(CHAPTER5_RESULTS / "metrics" / "chapter5_feature_selection_model_comparison.csv")


@app.get("/research/chapter5/predictions")
def chapter5_predictions(limit: int = 100) -> list[dict]:
    path = CHAPTER5_RESULTS / "predictions" / "do_10min_predictions.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No 10-minute predictions found. Run Chapter 5 experiment first.")
    df = pd.read_csv(path).head(limit)
    return df.to_dict(orient="records")


@app.get("/research/chapter5/feature_selection")
def chapter5_feature_selection() -> dict:
    fs_dir = CHAPTER5_RESULTS / "feature_selection"
    report_path = fs_dir / "final_feature_selection_report.json"
    if report_path.exists():
        return _read_json(report_path)
    return {
        "rfe_features": _read_csv_records(fs_dir / "rfe_selected_features.csv") if (fs_dir / "rfe_selected_features.csv").exists() else [],
        "final_features": _read_csv_records(fs_dir / "final_rfe_cfs_features.csv") if (fs_dir / "final_rfe_cfs_features.csv").exists() else [],
    }


@app.get("/research/combined/summary")
def combined_summary() -> dict:
    return _read_json(COMBINED_RESULTS / "results_summary.json")
