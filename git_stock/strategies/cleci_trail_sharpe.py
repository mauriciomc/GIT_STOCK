"""CleciTrailSharpe — same CleciTrail mechanics (buy near the trailing low, sell on the
band / trailing-stop / hard-stop), but hyperopted for **risk-adjusted return** instead of
the main bake-off's blended return/Sharpe/drawdown "recipe" objective.

Kept as its own registered strategy (rather than just re-scoring `clecitrail`'s search)
so its Sharpe/Sortino-optimised config is a distinct, comparable row in the bake-off
tables — see scripts/hyperopt_sharpe_sortino.py, objective "sharpe_sortino" in
git_stock/optimize.py.
"""
from __future__ import annotations

from .cleci_trail import CleciTrail


class CleciTrailSharpe(CleciTrail):
    pass
