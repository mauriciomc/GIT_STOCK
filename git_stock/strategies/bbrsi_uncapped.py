"""BbRsiUncapped - BbRsi with its stop-loss/take-profit guards removed.

BbRsi never set its own stoploss_pct/take_profit_pct, so it was silently
inheriting BacktestConfig's generic defaults (5% stop / 100% target). That
showed up directly in its trade log: every winning trade capped at EXACTLY
+98.6% (the 100% target minus realistic-cost friction) regardless of how much
further the stock actually ran afterward - GE, REGN, AAPL, NFLX and several
other multi-year winners were all cut off at the same ceiling. Every losing
trade clustered at the same -5.66% (the 5% stop minus friction) too.

This variant pins both wide open (matching the pattern already used for
CleciRangeCross) so the ONLY exit is BbRsi's own signal - price above the
upper Bollinger band or crossing below the midline, combined with RSI crossing
below rsi_sell. Lets winners actually ride the trend instead of getting capped
at a number that was never chosen for a reason, at the cost of also letting
losers run further before the signal (rather than a stop) gets them out.
"""
from __future__ import annotations

from .bbrsi import BbRsi


class BbRsiUncapped(BbRsi):
    # no exit is silently applied by the engine unless explicitly pinned open
    stoploss_pct = 99.0
    take_profit_pct = 10000.0
