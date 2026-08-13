"""Tests for reproducibility utilities."""

from __future__ import annotations

from src.reproducibility import collect_environment_versions, set_global_seeds


def test_set_global_seeds_runs():
    set_global_seeds(123)


def test_environment_versions():
    versions = collect_environment_versions()
    assert "python" in versions
    assert "numpy" in versions
