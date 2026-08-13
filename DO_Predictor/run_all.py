#!/usr/bin/env python
"""Main entry point for thesis research experiments.

Usage:
    python run_all.py              # runs Chapter 5 only (Chapter 3 requires dataset)
    python run_all.py --all        # Chapter 3 + Chapter 5
    python run_all.py --chapter3   # Chapter 3 only
    python run_all.py --chapter5   # Chapter 5 only
    python run_all.py --fast       # smoke test mode (NOT thesis results)
    python run_all.py --full       # full thesis mode (default when --all/--chapter*)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.research_config import FAST_MODE, FULL_MODE, apply_mode
from experiments.run_experiments import run_experiments
from src.reproducibility import save_environment_versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DO_Predictor thesis research experiments.")
    parser.add_argument("--fast", action="store_true", help="FAST_MODE smoke testing (not thesis results).")
    parser.add_argument("--full", action="store_true", help="FULL_MODE thesis experiments.")
    parser.add_argument("--chapter3", action="store_true", help="Run Chapter 3 pH experiment only.")
    parser.add_argument("--chapter5", action="store_true", help="Run Chapter 5 DO experiment only.")
    parser.add_argument("--all", action="store_true", help="Run both Chapter 3 and Chapter 5.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.fast:
        apply_mode(fast=True)
        print("FAST_MODE enabled — results are for pipeline testing only, NOT thesis reporting.")
    else:
        apply_mode(full=True)
        print("FULL_MODE enabled — complete thesis experiments.")

    run_ch3 = args.chapter3 or args.all
    run_ch5 = args.chapter5 or args.all

    # Default: Chapter 5 only (Chapter 3 requires separate dataset file)
    if not (run_ch3 or run_ch5):
        run_ch5 = True
        print("Default: running Chapter 5 only. Use --all or --chapter3 for Chapter 3.")

    save_environment_versions()
    results = run_experiments(chapter3=run_ch3, chapter5=run_ch5)

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    if results.get("chapter3"):
        c3 = results["chapter3"]
        print(f"Chapter 3: {c3.get('status')}")
        if c3.get("successful_models"):
            print("  Successful:", ", ".join(c3["successful_models"]))
        if c3.get("failed_models"):
            print("  Failed:", ", ".join(c3["failed_models"]))
        if c3.get("error"):
            print("  Error:", c3["error"])
    if results.get("chapter5"):
        c5 = results["chapter5"]
        print(f"Chapter 5: {c5.get('status')}")
        if c5.get("final_proposed_model"):
            print(f"  Final model: {c5['final_proposed_model']}")
        if c5.get("failed_models"):
            print(f"  Failed runs: {len(c5['failed_models'])}")
        if c5.get("error"):
            print("  Error:", c5["error"])
    print(f"\nResults directory: {PROJECT_ROOT / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
