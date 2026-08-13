"""Run thesis research experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.research_config import CHAPTER3_RESULTS, CHAPTER5_RESULTS, COMBINED_RESULTS
from experiments.chapter3_ph import run_chapter3
from experiments.chapter5_do import run_chapter5
from src.results_export import build_output_manifest, write_excel_sheets, write_json, write_text_summary


def run_experiments(chapter3: bool = True, chapter5: bool = True) -> dict:
    results = {"chapter3": None, "chapter5": None}
    if chapter3:
        results["chapter3"] = run_chapter3()
    if chapter5:
        results["chapter5"] = run_chapter5()
    _write_combined_results(results)
    return results


def _write_combined_results(results: dict) -> None:
    lines = ["DO_Predictor Research Results Summary", "=" * 40]
    master_rows = []

    if results.get("chapter3"):
        c3 = results["chapter3"]
        lines.append(f"Chapter 3 status: {c3.get('status')}")
        if c3.get("best_model"):
            lines.append(f"Best pH model (by RMSE): {c3['best_model']}")
        if c3.get("error"):
            lines.append(f"Chapter 3 error: {c3['error']}")

    if results.get("chapter5"):
        c5 = results["chapter5"]
        lines.append(f"Chapter 5 status: {c5.get('status')}")
        if c5.get("final_proposed_model"):
            lines.append(f"Final proposed DO model: {c5['final_proposed_model']}")
            fm = c5.get("final_metrics", {})
            lines.append(
                f"Final test metrics: MAE={fm.get('MAE')} RMSE={fm.get('RMSE')} MAPE={fm.get('MAPE')} R2={fm.get('R2')}"
            )
        if c5.get("error"):
            lines.append(f"Chapter 5 error: {c5['error']}")

    write_text_summary(COMBINED_RESULTS / "results_summary.txt", lines)
    write_json(COMBINED_RESULTS / "results_summary.json", results)
    manifest = build_output_manifest(CHAPTER3_RESULTS, CHAPTER5_RESULTS)
    write_json(COMBINED_RESULTS / "output_manifest.json", manifest)

    import pandas as pd

    if results.get("chapter3") and (CHAPTER3_RESULTS / "metrics" / "chapter3_model_comparison.csv").exists():
        master_rows.append(pd.read_csv(CHAPTER3_RESULTS / "metrics" / "chapter3_model_comparison.csv").assign(Chapter="3"))
    if results.get("chapter5") and (CHAPTER5_RESULTS / "metrics" / "chapter5_feature_selection_model_comparison.csv").exists():
        master_rows.append(
            pd.read_csv(CHAPTER5_RESULTS / "metrics" / "chapter5_feature_selection_model_comparison.csv").assign(Chapter="5")
        )
    if master_rows:
        master = pd.concat(master_rows, ignore_index=True)
        master.to_csv(COMBINED_RESULTS / "master_results.csv", index=False)
        write_excel_sheets(COMBINED_RESULTS / "master_results.xlsx", {"Master": master})


if __name__ == "__main__":
    run_experiments()
