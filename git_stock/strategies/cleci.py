"""Cleci — the "buy near the trailing low, sell near the trailing high" band strategy.

This is the original idea, fixed:

* the old class kept ``highest_low`` / ``lowest_low`` as **class attributes** that
  ``trader.py`` never reset between tickers, so only the first symbol in the run could
  ever fire a signal.  Here the bands are computed per-call and vectorised.
* ``window`` is the old, unused ``config["timewindow"]`` — now a real parameter.
  ``use_rolling=False`` reproduces the (buggy) expanding-window behaviour for comparison.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import crossed_above, crossed_below, rolling_max, rolling_min
from ..strategy import Strategy


class Cleci(Strategy):
    timeframe = "1d"
    space = {
        "window": (20, 400, 200, 10),
        "entry_mult": (1.02, 1.20, 1.10, 0.01),
        "exit_mult": (0.80, 0.98, 0.90, 0.01),
        "use_rolling": [True, False],
        # exit_mult 0.95 -> sell when close comes within 5% of the trailing high,
        # 0.90 -> within 10%, ... 0.50 -> within 50%
        # which series the trailing high tracks: "low" (original) or "high"
        "high_band": ["low", "high"],
        # "band"  = mechanical exit (close crosses band_high * exit_mult)
        # "roi"   = close on a take-profit / stop-loss (manual-close proxy)
        # "trail" = close on a trailing stop (let winners run, X% off the high-water mark)
        "exit_mode": ["band", "roi", "trail", "band+trail"],
        "tp": (3.0, 40.0, 15.0, 1.0),
        "sl": (4.0, 25.0, 8.0, 1.0),
        # floored at 6%: a sub-6% trailing stop on daily bars just scalps one day's noise
        "trail": (6.0, 20.0, 10.0, 0.5),
    }

    def __init__(self, params=None):
        super().__init__(params)
        mode = self.p["exit_mode"]
        if mode == "roi":
            self.take_profit_pct = float(self.p["tp"])
            self.stoploss_pct = float(self.p["sl"])
        elif mode in ("trail", "band+trail"):
            self.trailing_stop_pct = float(self.p["trail"])
            self.stoploss_pct = float(self.p["sl"])

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        w = int(self.p["window"])
        hi_src = df["high"] if self.p.get("high_band", "low") == "high" else df["low"]
        if self.p["use_rolling"]:
            df["band_low"] = rolling_min(df["low"], w)
            df["band_high"] = rolling_max(hi_src, w)
        else:
            df["band_low"] = df["low"].expanding().min()
            df["band_high"] = hi_src.expanding().max()
        # The actual decision thresholds entry_signal/exit_signal compare `close`
        # against - band_low/band_high alone are NOT what triggers a trade, only
        # these multiplied versions are. Stored as their own columns so charts (or
        # anything else inspecting `prepare()`'s output) can show the real trigger
        # levels instead of the raw bands, which sit 16-30% away from where a
        # signal actually fires and make the chart look like it disagrees with
        # its own entry/exit markers.
        df["entry_threshold"] = df["band_low"] * self.p["entry_mult"]
        df["exit_threshold"] = df["band_high"] * self.p["exit_mult"]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return crossed_below(df["close"], df["band_low"] * self.p["entry_mult"])

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        if self.p["exit_mode"] in ("roi", "trail"):
            return pd.Series(False, index=df.index)     # exits come from tp/sl or the trailing stop
        # "band" or "band+trail" — sell when close comes within (1-exit_mult) of the trailing high
        return crossed_above(df["close"], df["band_high"] * self.p["exit_mult"])

    def score(self, df: pd.DataFrame) -> pd.Series:
        # room left to the recent high-water mark, as a fraction of price — a name just
        # off its trailing low scores high (lots of room to mean-revert back up); a name
        # that's already run up near band_high scores near zero (little left to gain).
        # Same units for every symbol/day, so open positions and new candidates are
        # directly comparable for the backtest's rotation logic.
        return df["band_high"] / df["close"] - 1.0
