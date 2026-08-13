"""Export research results to CSV, Excel, JSON, and manifest files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_excel_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                safe_name = name[:31]
                frame.to_excel(writer, sheet_name=safe_name, index=False)
    except ImportError:
        csv_path = path.with_suffix(".csv")
        pd.concat(
            [frame.assign(_sheet=name) for name, frame in sheets.items() if not frame.empty],
            ignore_index=True,
        ).to_csv(csv_path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_text_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_output_manifest(chapter3_dir: Path, chapter5_dir: Path) -> dict[str, Any]:
    def _collect(root: Path, subdirs: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for sub in subdirs:
            folder = root / sub
            out[sub] = sorted(str(p.relative_to(root)).replace("\\", "/") for p in folder.rglob("*") if p.is_file())
        return out

    return {
        "chapter3": _collect(chapter3_dir, ["metrics", "predictions", "figures", "models", "tables", "data"]),
        "chapter5": _collect(
            chapter5_dir, ["metrics", "predictions", "figures", "models", "tables", "data", "feature_selection"]
        ),
    }
