# DO_Predictor

Dissolved Oxygen forecasting benchmark on the Brisbane water-quality buoy dataset. Built
as a thesis-grade comparative model benchmark: every model is trained on a strict
chronological train/val/test split, evaluated the same way, and reported against both a
persistence floor and a published public baseline.

## Dataset

Kaggle "Water Quality Monitoring Dataset" (Brisbane buoy), owner/slug
`downshift/water-quality-monitoring-dataset`, file `brisbane_water_quality.csv`.
Downloaded automatically via `kagglehub` (no Kaggle API key needed for this public
dataset - `src/data_loader.py` handles it).

The raw file has 20 columns, including `Timestamp`, `Record number`, and nine sensor
readings each paired with a `<column> [quality]` instrument-code column. This project is
**Dissolved-Oxygen-only**: the `pH` and `pH [quality]` columns are dropped immediately
at load time (`src/data_loader.py`) and never reach validation, cleaning, feature
engineering, or any model - pH is present in the source file but is not used anywhere in
this system.

### What the raw data actually looks like (see `notebooks/01_eda.ipynb`)

- **The `[quality]` columns are real, not empty.** All of them (for the sensors this
  project uses) are populated with actual instrument codes (dominant code `1020` =
  "good"; a small number of `2010`/`1220` codes appear too). None were dropped - see
  `src/validation.py` / `QualityReport.quality_columns_kept`.
- **`Timestamp` is not a safe sort key.** It has 280 exact duplicate values and 2
  backward jumps. `Record number` is strictly increasing with zero gaps across all
  30,894 rows and is used as the canonical row order everywhere in the codebase.
- **The dataset is commonly described as 30-minute-interval data. It isn't.** Ordered
  by `Record number`, the modal spacing between consecutive readings is **10 minutes**,
  with a secondary mode at 30 minutes and two multi-day sensor-outage gaps. `RESAMPLE_FREQ`
  in `src/config.py` is set to `10min` to match reality.

## Baseline being compared against

A public Kaggle notebook on this same dataset reports SARIMAX MAE **0.60** and Prophet
MAE **0.40** for Dissolved Oxygen forecasting. Both are cited in `src/config.py`
(`BASELINE_SARIMAX_MAE`, `BASELINE_PROPHET_MAE`) and every result in
`experiments/results/model_comparison.json` is compared against them automatically.

Our own SARIMAX/Prophet/ARIMA implementations are trained under a **harder** protocol
than most public benchmarks: a single-shot forecast across the entire held-out test
window (roughly 10 days at 10-minute cadence), not a walk-forward one-step evaluation.
Under that protocol classical statistical forecasts drift over the horizon and can
underperform the cited numbers - this is a real, expected property of long-horizon
static forecasting, not a bug (verified directly - see "Methodology notes" below).
Lag-informed ML/DL models, by contrast, get fresh lagged sensor context at every step,
which is why they beat the baseline by a wide margin.

## Project structure

```
data/{raw,processed}/          raw CSV cache + cleaned/resampled parquet
src/                           pipeline modules (see below)
src/models/                    base_model.py + registry.py + one file per model
experiments/                   run_ablation.py + results/*.json
notebooks/                     01_eda, 02_ablation_analysis, 03_model_comparison
backend/                       FastAPI service (app.py, schemas.py)
frontend/                      Streamlit demo app
trained_models/                saved model artifacts (.joblib / .keras)
reports/                       exported figures for the thesis
tests/                         pytest suite (pipeline sanity + per-model adapter checks)
```

## Pipeline (`src/pipeline.py`)

`load -> validate -> clean -> smooth -> features -> train -> eval`

1. **`data_loader.py`** - downloads/caches the CSV, parses `Timestamp`, drops the `pH`
   columns (this system is DO-only).
2. **`validation.py`** - read-only integrity report: missingness, duplicate
   timestamps, `Record number` vs `Timestamp` ordering, sampling cadence, physically
   impossible values, quality-column usefulness. Nothing is mutated here.
3. **`cleaning.py`** - nulls out sensor readings flagged bad by their quality code,
   resamples onto a fixed 10-minute grid (this is also what resolves the 280 duplicate
   timestamps - they land in the same bin and get averaged), interpolates gaps up to
   60 minutes (longer gaps, like the two real sensor outages, are left as NaN rather
   than papered over), and flags (not drops) rolling-z-score outliers.
4. **`smoothing.py`** - adds moving-average and Savitzky-Golay smoothed columns
   alongside the raw signal (both are available as features, not a replacement).
5. **`feature_engineering.py`** - lags, rolling mean/std, rate of change, cyclical
   hour/day-of-year encodings, austral season, and the shifted target column.
6. **`pipeline.py`** - wires it together, builds four progressively-processed "stages"
   (`cleaned` -> `smoothed` -> `lag_features` -> `full_features`) for the ablation
   study, and does a strict chronological 70/15/15 train/val/test split (no shuffling).
   ARIMA/SARIMAX/Prophet and the six sequence DL models additionally get restricted to
   the single longest gap-free contiguous block of the resampled series, since a
   forecast or sliding window spanning the multi-day outage would silently combine
   unrelated time periods.

## Model registry (`src/models/`)

Every model implements `fit` / `predict` / `evaluate` / `get_params` via
`base_model.BaseModel`. Models whose natural input shape differs from a flat
lag-feature matrix get a thin adapter:

| Adapter | Used by |
|---|---|
| `SklearnRegressorAdapter` | Linear/Ridge/Lasso, SVR, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost, KNN |
| `PersistenceModel` | persistence |
| `StatsForecastAdapter` | ARIMA, SARIMAX |
| `ProphetAdapter` | Prophet |
| `SequenceModelAdapter` | LSTM, GRU, CNN-LSTM, BiLSTM, Transformer, 1D-CNN (TensorFlow/Keras) |

**Statistical**: Persistence, Linear Regression, Ridge, Lasso, ARIMA, SARIMAX, Prophet
**Classical ML**: SVR, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, KNN
**Deep Learning**: LSTM, GRU, CNN-LSTM, BiLSTM, Transformer encoder, 1D-CNN

Every model was individually shape/output-verified in `tests/test_models.py` before
being wired into the registry (`src/models/registry.py`) - all 20 pass.

## Methodology notes worth knowing before reading the results

- **SVR is capped to the most recent 5,000 training rows.** RBF-kernel SVR is
  O(n^2)-O(n^3); on ~15k rows it doesn't finish in reasonable time. The cap keeps the
  most recent (still chronologically valid) slice of training data - documented in
  `src/models/svr_model.py`.
- **SARIMAX has no seasonal term.** A daily seasonal component at 144 steps (10-minute
  cadence) makes the state space intractable to fit. Cyclical hour/day-of-year columns
  are included in exog instead - see `src/config.get_exog_columns`.
- **Prophet uses `growth='flat'` and `yearly_seasonality=False`.** Prophet's default
  linear trend extrapolates whatever local slope existed near the end of training
  across the entire multi-week test horizon and diverges to physically nonsensical
  values (verified directly during development - trend blew up from ~7 mg/L to ~65
  mg/L). DO oscillates, it doesn't trend, so flat growth is the correct model.
  Yearly seasonality is disabled because the longest contiguous data block (~2 months)
  isn't remotely enough to identify an annual cycle.
- **ARIMA/SARIMAX/Prophet forecast the entire test horizon in one shot** (`get_forecast(steps=len(test))`
  / Prophet's native future-dataframe prediction), not walk-forward. Lag-feature ML/DL
  models predict one step ahead using the actual lagged history available at each row.
  This is a genuine, documented difference in how each model family is conventionally
  deployed, not an inconsistency - and it's exactly why the lag-informed models win by
  such a wide margin.

## Results

Full numbers: `experiments/results/model_comparison.json` (headline table) and
`experiments/results/stage_metrics.json` (preprocessing-stage ablation). See
`notebooks/03_model_comparison.ipynb` for the formatted table/charts and
`notebooks/02_ablation_analysis.ipynb` for the stage-wise breakdown.

All 20 models trained successfully (20/20, zero silent drops) on the `full_features`
stage, chronological 70/15/15 split.

### Dissolved Oxygen - test set, sorted by MAE

| Rank | Model | MAE | RMSE | R² | vs SARIMAX baseline (0.60) | vs Prophet baseline (0.40) |
|---|---|---|---|---|---|---|
| 1 | linear_regression | 0.0966 | 0.1328 | 0.932 | **+83.9%** | **+75.9%** |
| 2 | ridge | 0.0969 | 0.1331 | 0.932 | +83.8% | +75.8% |
| 3 | lasso | 0.0998 | 0.1365 | 0.928 | +83.4% | +75.0% |
| 4 | lightgbm | 0.1161 | 0.1566 | 0.905 | +80.7% | +71.0% |
| 5 | xgboost | 0.1174 | 0.1582 | 0.904 | +80.4% | +70.7% |
| 6 | random_forest | 0.1193 | 0.1623 | 0.898 | +80.1% | +70.2% |
| 7 | gradient_boosting | 0.1283 | 0.1721 | 0.886 | +78.6% | +67.9% |
| 8 | catboost | 0.1363 | 0.1819 | 0.872 | +77.3% | +65.9% |
| 9 | svr | 0.1766 | 0.2232 | 0.808 | +70.6% | +55.9% |
| 10 | persistence | 0.2029 | 0.2785 | 0.701 | +66.2% | +49.3% |
| 11 | transformer | 0.3317 | 0.4329 | 0.533 | +44.7% | +17.1% |
| 12 | bilstm | 0.3443 | 0.5824 | 0.154 | +42.6% | +13.9% |
| 13 | cnn_lstm | 0.3994 | 0.5178 | 0.331 | +33.4% | +0.1% |
| 14 | prophet (baseline) | 0.4600 | 0.5889 | 0.129 | +23.3% | 0% |
| 15 | knn | 0.5222 | 0.6264 | -0.513 | +13.0% | -30.6% |
| 16 | arima | 0.6058 | 0.7622 | -0.460 | -1.0% | -51.5% |
| 17 | gru | 0.6197 | 1.3384 | -3.467 | -3.3% | -54.9% |
| 18 | lstm | 0.7389 | 1.5552 | -5.031 | -23.1% | -84.7% |
| 19 | cnn1d | 1.2982 | 1.9836 | -8.811 | -116.4% | -224.6% |
| 20 | sarimax (baseline) | 1.3030 | 1.5102 | -4.731 | -117.2% | -225.8% |

**Best model beats the cited SARIMAX baseline by 84% and the Prophet baseline by 76%
(linear regression, MAE 0.097 vs 0.60 / 0.40).**

### Honest limitations (disclosed, not dropped)

- **Every one of the 20 registered models trained and evaluated successfully -
  nothing failed outright.** But several results are genuinely bad and are reported
  as such rather than hidden:
- **CNN1D and SARIMAX are the weakest models** (MAE 1.30 each, both worse than the
  persistence floor). CNN1D's architecture ends in `GlobalAveragePooling1D`, which
  collapses the entire 4-hour lookback window into a single average - throwing away
  exactly the "most recent value" signal that makes persistence and lag-feature models
  so effective on this near-random-walk series.
- **Linear models beat every classical/DL model.** At a 10-minute forecast horizon, DO
  is close to a smooth random walk, which is exactly the regime where a simple linear
  combination of recent lags wins - this isn't a failure of the more complex models,
  it's a real property of the forecasting task.
- **Deep learning models underperform classical ML overall**, most plausibly because
  they only get the longest contiguous gap-free block for training (~6,900 rows) versus
  ~14,800 rows available to the lag-matrix models, on top of the inherent difficulty of
  out-learning a near-linear signal with a few thousand samples.
- **Our own ARIMA/SARIMAX/Prophet underperform the cited public baseline.** See
  "Methodology notes" above - different (harder) evaluation protocol, not a
  broken implementation (verified directly during development).

## Install & run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# full pipeline sanity + all 20 model adapters
pytest tests/ -v

# full ablation (stage sweep + full registry) - takes a while
python -m experiments.run_ablation

# API
uvicorn backend.app:app --reload

# demo UI
streamlit run frontend/streamlit_app.py
```

## API (`backend/app.py`)

| Endpoint | Purpose |
|---|---|
| `POST /upload` | Upload a CSV, get back a quality report |
| `POST /train` | Train one model (or the whole registry), save to `trained_models/` |
| `GET /metrics` | Pull stored comparison metrics |
| `GET /predict` | Latest prediction + recent actual-vs-predicted + low-DO threshold flag |
| `GET /history` | Every training run this process has done |
| `GET /quality_report` | Data-quality report for the currently loaded dataset |
| `GET /download_report` | Zip of everything in `reports/` |

## Frontend (`frontend/streamlit_app.py`)

Upload a CSV (or use the bundled dataset), pick a model set (quick 6-model subset or
the full 20-model registry), run the pipeline, and get a sortable comparison table with
the best model highlighted, an MAE bar chart against the cited baseline, and a low-DO
threshold flag on the latest prediction. Monitoring flag only - no hardware control.

---

# Thesis Research Pipeline (Chapter 3 & Chapter 5)

This repository now includes a **modular thesis research system** in addition to the legacy DO benchmark above.

## Research experiments

| Experiment | Target | Contribution | Dataset path |
|---|---|---|---|
| **Chapter 3** | pH | ARIMA + LSTM hybrid (`y_hat = y_hat_ARIMA + e_hat_LSTM`) | `data/raw/chapter3_ph_dataset.csv` (**must be provided**) |
| **Chapter 5** | DO | RFE → CFS → RFE+CFS feature selection, 10-min forecast | `data/raw/chapter5_do_dataset.csv` (bundled Brisbane copy) |

Chapter 3 and Chapter 5 are **independent**. The Chapter 5 dataset is **not** substituted for Chapter 3.

## Installation

```bash
cd DO_Predictor
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
```

**Required for FULL_MODE:** TensorFlow (LSTM/GRU/Transformer), statsmodels (ARIMA), scikit-learn, openpyxl (Excel export).

## Configuration

All research settings live in `config/research_config.py`:

| Setting | Default | Notes |
|---|---|---|
| `SEED` | 42 | Reproducibility |
| `CHAPTER3_DATA_PATH` | `data/raw/chapter3_ph_dataset.csv` | Must exist for Chapter 3 |
| `CHAPTER5_DATA_PATH` | `data/raw/chapter5_do_dataset.csv` | Brisbane DO validation set |
| `TRAIN/VAL/TEST_FRACTION` | 0.70 / 0.15 / 0.15 | Chronological split |
| `DO_FORECAST_MINUTES` | 10 | 10-minute-ahead DO target |
| `LOW_DO_THRESHOLD` | `None` | **Must be configured from study criterion** |
| `FAST_MODE` / `FULL_MODE` | runtime via `run_all.py` | FAST = smoke test only |

## Running experiments

```bash
# Default: Chapter 5 only (Chapter 3 requires dataset)
python run_all.py

# Smoke / integration test (NOT thesis results)
python run_all.py --fast --chapter5

# Full thesis experiments
python run_all.py --full --chapter5
python run_all.py --full --chapter3   # requires chapter3_ph_dataset.csv
python run_all.py --full --all
```

## Results directory

```
results/
  chapter3_ph/   {data, models, metrics, predictions, figures, tables, logs}
  chapter5_do/   {data, models, metrics, predictions, feature_selection, figures, tables, logs}
  combined/      master_results.csv/xlsx, results_summary.json/txt, output_manifest.json
```

Key outputs:
- `chapter3_results.xlsx`, `chapter5_results.xlsx`, `master_results.xlsx`
- `do_10min_predictions.csv`, `hybrid_predictions.csv`
- `chapter5_feature_selection_model_comparison.csv`
- Publication figures (PNG 300 DPI + PDF) in `figures/`

## Streamlit research dashboard

```bash
streamlit run frontend/streamlit_app.py
```

Select **Research Dashboard** in the sidebar, then choose Chapter 3 or Chapter 5. All metrics and figures are loaded from `results/` — nothing is hard-coded.

Legacy **DO Benchmark** mode preserves the original 20-model comparison UI.

## Research API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /research/status` | Which result files exist |
| `GET /research/chapter3/summary` | Chapter 3 experiment summary |
| `GET /research/chapter3/comparison` | Model comparison table |
| `GET /research/chapter5/summary` | Chapter 5 experiment summary |
| `GET /research/chapter5/comparison` | Feature-set × model comparison |
| `GET /research/chapter5/predictions` | 10-minute DO predictions |
| `GET /research/chapter5/feature_selection` | RFE+CFS report |
| `GET /research/combined/summary` | Combined results JSON |

## Testing

```bash
pytest -q --ignore=venv
python -m compileall config experiments src tests run_all.py
```

## Troubleshooting

| Issue | Action |
|---|---|
| Chapter 3 fails immediately | Place `chapter3_ph_dataset.csv` at configured path |
| LOW-DO alerts empty | Set `LOW_DO_THRESHOLD` in `config/research_config.py` |
| FAST_MODE slow | Expected — use `--fast` for pipeline validation only |
| TensorFlow warnings on Windows | CPU-only is normal; GPU requires WSL2 or DirectML |

## What you must provide

1. **`data/raw/chapter3_ph_dataset.csv`** — your Chapter 3 pH experiment dataset
2. **`LOW_DO_THRESHOLD`** — aquaculture hypoxia criterion (if alerts required)
3. Optional: `WARNING_DO_THRESHOLD`, custom dataset paths in config
