"""Load a per-sector strategy config: each sector gets its OWN fully independent
strategy instance (different indicator params, not just a stake-size weight),
alongside its own stake multiplier - both hyperopted together and stored in one
JSON blob, e.g. config/sector_params_swingtrend.json:

    {"_comment": "...",
     "tech": {"lookback": 12, "reversal_pct": 18.0, ..., "weight": 1.8},
     "energy": {"lookback": 30, ..., "weight": 0.6},
     ...}

Use with git_stock.backtest.run(), which accepts EITHER a single Strategy
(applied to every symbol, existing behaviour) OR a dict[sector_name, Strategy]
for this per-sector mode - see backtest.run()'s docstring.
"""
from __future__ import annotations

import json

from .strategy import Strategy


def load_sector_strategies(path: str, strategy_cls: type[Strategy]) -> tuple[dict[str, Strategy], dict[str, float]]:
    """Returns (per_sector_strategies, sector_weights) - pass the first as `strategy`
    and the second as `BacktestConfig.sector_weights` to git_stock.backtest.run()."""
    raw = json.load(open(path))
    strategies, weights = {}, {}
    for sector, cfg in raw.items():
        if sector == "_comment":
            continue
        cfg = dict(cfg)
        weights[sector] = cfg.pop("weight", 1.0)
        strategies[sector] = strategy_cls(cfg)
    return strategies, weights
