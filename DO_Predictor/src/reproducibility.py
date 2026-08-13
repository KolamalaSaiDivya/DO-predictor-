"""Reproducibility: seed setting and environment version capture."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from config.research_config import COMBINED_RESULTS, SEED, set_python_hash_seed


def set_global_seeds(seed: int = SEED) -> None:
    """Set Python, NumPy, TensorFlow, and PYTHONHASHSEED."""
    set_python_hash_seed()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def collect_environment_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": _safe_version("numpy"),
        "pandas": _safe_version("pandas"),
        "scikit-learn": _safe_version("sklearn"),
        "statsmodels": _safe_version("statsmodels"),
        "tensorflow": _safe_version("tensorflow"),
        "xgboost": _safe_version("xgboost"),
        "matplotlib": _safe_version("matplotlib"),
        "seaborn": _safe_version("seaborn"),
        "openpyxl": _safe_version("openpyxl"),
        "scipy": _safe_version("scipy"),
    }
    return versions


def _safe_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not installed"


def save_environment_versions(output_path: Path | None = None) -> Path:
    output_path = output_path or (COMBINED_RESULTS / "environment_versions.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    versions = collect_environment_versions()
    lines = [f"{key}: {value}" for key, value in versions.items()]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print("Environment versions saved to", output_path)
    for line in lines:
        print(" ", line)
    return output_path
