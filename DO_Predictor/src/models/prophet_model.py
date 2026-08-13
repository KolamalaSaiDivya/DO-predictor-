from __future__ import annotations

from src.models.base_model import ProphetAdapter


def build_prophet(target_col: str) -> ProphetAdapter:
    # growth='flat': default linear trend extrapolates whatever slope was
    # near the end of training clean out across the whole test horizon (weeks
    # ahead) and blows up to nonsense values - DO doesn't have a real secular
    # trend, it oscillates. yearly_seasonality off because the longest
    # contiguous block is ~2 months, nowhere near enough to identify a yearly
    # cycle (Prophet will happily fit one anyway and extrapolate garbage).
    return ProphetAdapter(
        name="prophet",
        growth="flat",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
    )
