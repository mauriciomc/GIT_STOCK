"""SwingTrendTuned - SwingTrend with its own hyperopt-searchable take-profit,
instead of silently inheriting BacktestConfig's generic 100% default.

Found via direct inspection of SwingTrend's NVDA trade log: every winning trade
capped at EXACTLY +98.6% (the 100% target minus realistic-cost friction) no
matter how far the stock kept running afterward - e.g. a 2024-01-22 entry closed
at +98.6% while NVDA continued well past that level for another year. SwingTrend
sets its own trailing_stop_pct and stoploss_pct but never touches take_profit_pct,
so every trade silently inherited the config-level default. This variant searches
tp properly (wide range, since the whole point of a trend-following entry is not
capping the rare huge winner) instead of leaving it fixed at 100%.
"""
from __future__ import annotations

from .swing_trend import SwingTrend


class SwingTrendTuned(SwingTrend):
    space = {**SwingTrend.space,
             "tp": (30.0, 400.0, 150.0, 10.0)}

    def __init__(self, params=None):
        super().__init__(params)
        self.take_profit_pct = float(self.p["tp"])
