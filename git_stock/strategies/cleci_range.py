"""CleciRange — same "buy near the trailing low, sell near the trailing high" idea as
Cleci, but with the thresholds expressed as a % of the trailing range instead of
independent multipliers on each end.

Cleci computes buy/sell levels as (lowest_low * entry_mult) and (highest_high *
exit_mult) - two multipliers applied to two DIFFERENT base values. For a stock whose
trailing high isn't at least entry_mult/exit_mult times its trailing low, those two
levels cross (buy trigger ends up ABOVE the sell trigger) - confirmed on MMM, whose
120-day range (33%) is narrower than the ~51% Cleci's own default multipliers need to
stay correctly ordered. The mechanical exit becomes nearly unreachable in that case
since price would need to fall back below sell_threshold before it could ever "cross
above" it again.

CleciRange fixes this by construction, the classic Stochastic %K idea:

    rng = highest_high - lowest_low
    buy_threshold  = lowest_low + rng * (buy_pct  / 100)   # e.g. 16 -> bottom 16% of range
    sell_threshold = lowest_low + rng * (sell_pct / 100)   # e.g. 77 -> top 23% of range

Both thresholds are now percentages of the SAME range, so sell_threshold > buy_threshold
is guaranteed for any stock as long as buy_pct < sell_pct - no dependency on how much
the stock actually moved.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import crossed_above, crossed_below, rolling_max, rolling_min
from ..strategy import Strategy


class CleciRange(Strategy):
    timeframe = "1d"
    space = {
        "window": (20, 400, 200, 5),
        "buy_pct": (5.0, 40.0, 16.0, 0.5),
        "sell_pct": (60.0, 95.0, 77.0, 0.5),
        "high_band": ["low", "high"],
        "exit_mode": ["band", "roi", "trail", "band+trail"],
        "tp": (3.0, 40.0, 15.0, 0.5),
        "sl": (4.0, 25.0, 8.0, 0.5),
        "trail": (6.0, 20.0, 10.0, 0.25),
        # trail only arms once unrealised gain reaches this % above entry - 0 recovers
        # the old always-armed behaviour, so hyperopt can find "no activation offset
        # helps" on its own rather than it being forced.
        "trail_activation": (0.0, 15.0, 0.0, 0.5),
    }

    def __init__(self, params=None):
        super().__init__(params)
        if self.p["buy_pct"] >= self.p["sell_pct"]:
            raise ValueError(
                f"buy_pct ({self.p['buy_pct']}) must be < sell_pct ({self.p['sell_pct']}) "
                "or the bands invert by construction"
            )
        mode = self.p["exit_mode"]
        if mode == "roi":
            self.take_profit_pct = float(self.p["tp"])
            self.stoploss_pct = float(self.p["sl"])
        elif mode in ("trail", "band+trail"):
            self.trailing_stop_pct = float(self.p["trail"])
            self.trailing_activation_pct = float(self.p["trail_activation"])
            self.stoploss_pct = float(self.p["sl"])

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        w = int(self.p["window"])
        hi_src = df["high"] if self.p.get("high_band", "low") == "high" else df["low"]
        df["band_low"] = rolling_min(df["low"], w)
        df["band_high"] = rolling_max(hi_src, w)

        rng = df["band_high"] - df["band_low"]
        df["entry_threshold"] = df["band_low"] + rng * (self.p["buy_pct"] / 100.0)
        df["exit_threshold"] = df["band_low"] + rng * (self.p["sell_pct"] / 100.0)
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return crossed_below(df["close"], df["entry_threshold"])

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        if self.p["exit_mode"] in ("roi", "trail"):
            return pd.Series(False, index=df.index)
        return crossed_above(df["close"], df["exit_threshold"])

    def score(self, df: pd.DataFrame) -> pd.Series:
        # room left to the sell threshold, as a fraction of price - same role as
        # Cleci.score, just measured against the real trigger level instead of the
        # raw band_high.
        return df["exit_threshold"] / df["close"] - 1.0
