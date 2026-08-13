"""DO(t+1) = DO(t) - the floor every other model in the registry has to clear."""

from __future__ import annotations

from src.models.base_model import PersistenceModel


def build_persistence(target_col: str) -> PersistenceModel:
    return PersistenceModel(name="persistence", target_col=target_col)
