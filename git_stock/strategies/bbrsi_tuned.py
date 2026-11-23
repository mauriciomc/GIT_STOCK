"""BbRsiTuned - BbRsi with its own hyperopt-searchable stop-loss/take-profit,
instead of silently inheriting BacktestConfig's generic 5%/100% defaults.

Found via direct inspection of BbRsi's trade log: every winning trade capped at
EXACTLY +98.6% (the 100% target minus realistic-cost friction) regardless of how
much further the stock ran afterward, and losers clustered at exactly -5.66% (the
5% stop minus friction) - the strategy's own Bollinger/RSI exit signal was rarely
what actually closed a trade. A follow-up test removing the stop/target entirely
(BbRsiUncapped) showed the opposite failure mode: positions held 5-10+ years
waiting for the signal, including a -59% loss held 5.6 years and a -45% loss held
6.8 years with no protection at all. This variant searches sl/tp properly instead
of picking either extreme.
"""
from __future__ import annotations

from .bbrsi import BbRsi


class BbRsiTuned(BbRsi):
    space = {**BbRsi.space,
             "sl": (10.0, 40.0, 20.0, 1.0),
             "tp": (20.0, 150.0, 60.0, 5.0)}

    def __init__(self, params=None):
        super().__init__(params)
        self.stoploss_pct = float(self.p["sl"])
        self.take_profit_pct = float(self.p["tp"])
