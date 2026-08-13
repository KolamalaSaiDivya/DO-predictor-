# Data Directory

## Required datasets

| File | Experiment | Description |
|------|------------|-------------|
| `raw/chapter3_ph_dataset.csv` | Chapter 3 | Multivariate pH forecasting dataset (must be provided) |
| `raw/chapter5_do_dataset.csv` | Chapter 5 | Brisbane River DO validation dataset |

## Chapter 5 dataset

`chapter5_do_dataset.csv` is the Brisbane water quality monitoring dataset with canonical column mapping (`Dissolved Oxygen` → `DO`). The bundled copy is derived from `brisbane_water_quality.csv`.

## Chapter 3 dataset

Place your Chapter 3 experimental dataset at `raw/chapter3_ph_dataset.csv`. The file must contain a numeric `pH` target column (aliases `ph`, `PH` are normalized automatically).

**Do not substitute the Chapter 5 dataset for Chapter 3.** If the Chapter 3 file is missing, Chapter 3 stops with a clear error and Chapter 5 can still run.

## Supported formats

CSV, XLSX, Parquet (via `src/research_data_loader.py`).
