"""CleciRangeBounce - a reclaim/breakout variant of CleciRangeRSI.

CleciRangeRSI buys while price is still falling (crosses BELOW entry_threshold -
catching the knife) and only optionally exits on a band cross. This variant
flips the entry logic to require STRENGTH before buying, and adds a second,
independent entry trigger off the top of the range:

Buy: EITHER
  - price bounces off or reclaims entry_threshold (the green/support line) - a
    same-candle wick rejection (low touches at/below the line, close finishes
    above it) OR a multi-candle crossover reclaim (closed below on a prior bar,
    closed back above now). A plain cross BELOW never buys.
  - OR price bounces off or reclaims exit_threshold (the red/resistance line) -
    same two patterns - i.e. a breakout-with-retest above the top of the range.
  AND RSI(period) > rsi_buy_min (same optional confirmation gate as CleciRangeRSI;
  0 recovers "no RSI filter").

Sell: price crosses BELOW exit_threshold (the mirror image of CleciRangeRSI's
crossed-ABOVE exit) AND RSI(period) < rsi_sell_max - once price that broke out
above resistance rolls back under it, or a reclaimed range gives back its gain,
get out. Unlike CleciRangeRSI, this rule is ALWAYS active regardless of
exit_mode - exit_mode only controls whether tp/sl/trail also apply as
additional, potentially earlier exits.

cooldown_bars: after ANY exit (band-cross sell, stoploss, trailing, take-profit)
the symbol sits out this many bars before a new entry is allowed - stops the
bounce/breakout triggers from immediately re-firing on the very next bar's
noise right after a stop-out.

Same range-percentage band construction as CleciRange/CleciRangeRSI (rolling
window, buy_pct/sell_pct of the SAME range - correct ordering guaranteed by
construction) and the same exit_mode/tp/sl/trail/trail_activation machinery,
so hyperopt can pick roi/trail/band+trail as an EARLIER exit on top of the
always-on band-cross sell.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import crossed_above, crossed_below, rolling_max, rolling_min, rsi
from ..strategy import Strategy


class CleciRangeBounce(Strategy):
    timeframe = "1d"
    space = {
        "window": (20, 400, 200, 5),
        "buy_pct": (5.0, 40.0, 16.0, 0.5),
        "sell_pct": (60.0, 95.0, 77.0, 0.5),
        "rsi_period": (7, 21, 14, 1),
        # 0/100 bounds recover "no RSI filter" - hyperopt can find "RSI doesn't
        # help" on its own if that's the honest answer
        "rsi_buy_min": (0.0, 50.0, 30.0, 0.5),
        "rsi_sell_max": (50.0, 100.0, 70.0, 0.5),
        "high_band": ["low", "high"],
        "exit_mode": ["band", "roi", "trail", "band+trail"],
        "tp": (3.0, 40.0, 15.0, 0.5),
        "sl": (4.0, 25.0, 8.0, 0.5),
        "trail": (6.0, 20.0, 10.0, 0.25),
        "trail_activation": (0.0, 15.0, 0.0, 0.5),
        # bars a symbol must sit out after ANY exit before it's allowed back in -
        # stops the bounce/breakout triggers from immediately re-firing on the
        # very next bar's noise right after a stop-out or a band-cross sell.
        "cooldown_bars": (0, 20, 2, 1),
    }

    def __init__(self, params=None):
        super().__init__(params)
        if self.p["buy_pct"] >= self.p["sell_pct"]:
            raise ValueError(
                f"buy_pct ({self.p['buy_pct']}) must be < sell_pct ({self.p['sell_pct']}) "
                "or the bands invert by construction"
            )
        self.cooldown_bars = int(self.p["cooldown_bars"])
        mode = self.p["exit_mode"]
        if mode == "roi":
            self.take_profit_pct = float(self.p["tp"])
            self.stoploss_pct = float(self.p["sl"])
        elif mode in ("trail", "band+trail"):
            self.trailing_stop_pct = float(self.p["trail"])
            self.trailing_activation_pct = float(self.p["trail_activation"])
            self.stoploss_pct = float(self.p["sl"])
        # always-on trail off the red line, independent of exit_mode: once the
        # position's high-water mark clears TODAY's exit_threshold by 6%, arm a
        # tight 3% trailing stop - lock in gains once price is clearly through
        # resistance, on top of whatever exit_mode's own tp/sl/trail adds.
        self.trailing_stop_pct = 3.0
        self.trailing_activation_pct = 6.0
        self.trail_activation_from = "exit_threshold"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        w = int(self.p["window"])
        hi_src = df["high"] if self.p.get("high_band", "low") == "high" else df["low"]
        df["band_low"] = rolling_min(df["low"], w)
        df["band_high"] = rolling_max(hi_src, w)

        rng = df["band_high"] - df["band_low"]
        df["entry_threshold"] = df["band_low"] + rng * (self.p["buy_pct"] / 100.0)
        df["exit_threshold"] = df["band_low"] + rng * (self.p["sell_pct"] / 100.0)
        df["rsi"] = rsi(df["close"], int(self.p["rsi_period"]))
        return df

    @staticmethod
    def _bounced_or_reclaimed(df: pd.DataFrame, level: pd.Series) -> pd.Series:
        same_bar_bounce = (df["low"] <= level) & (df["close"] > level)
        multi_bar_reclaim = crossed_above(df["close"], level)
        return same_bar_bounce | multi_bar_reclaim

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        green = self._bounced_or_reclaimed(df, df["entry_threshold"])
        red = self._bounced_or_reclaimed(df, df["exit_threshold"])
        return (green | red) & (df["rsi"] > self.p["rsi_buy_min"])

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        # crossing below the red line is ALWAYS a sell, regardless of exit_mode -
        # exit_mode only controls whether tp/sl/trail ALSO apply as additional,
        # potentially earlier exits, not whether this one fires at all.
        crossed = crossed_below(df["close"], df["exit_threshold"])
        return crossed & (df["rsi"] < self.p["rsi_sell_max"])

    def score(self, df: pd.DataFrame) -> pd.Series:
        return df["exit_threshold"] / df["close"] - 1.0
