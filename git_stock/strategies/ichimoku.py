"""Ichimoku cloud trend follower — a deliberately *slow* strategy: enter only when the
whole trend structure lines up, hold until the trend line (Kijun) breaks.

Rebuilt from the old ``src/trader.py`` ``strategy_ichi`` without its look-ahead bug
(the original used ``ha_close.shift(-26)`` as the chikou span, which reads the future).
Everything here is causal: the cloud is projected forward with ``.shift(base)``, and the
chikou confirmation is "price today vs price `base` bars ago".
"""
from __future__ import annotations

import pandas as pd

from ..indicators import crossed_below, heikin_ashi, rolling_max, rolling_min
from ..strategy import Strategy


class Ichimoku(Strategy):
    timeframe = "1d"
    space = {
        "conv": (5, 20, 9, 1),          # tenkan-sen period
        "base": (20, 60, 26, 2),        # kijun-sen period / cloud displacement
        "span_b": (40, 120, 52, 4),     # senkou span B period
        "use_ha": [True, False],        # smooth with heikin-ashi candles
        "exit_rule": ["tk_cross", "kijun_break", "cloud_loss"],
        "sl": (6.0, 30.0, 15.0, 1.0),   # protective hard stop %
        # 0 = off, else a *wide* trailing stop; sub-6% trails on daily bars degenerate
        # into 1-bar scalps (the trail=1% / 21k-trade ichimoku config)
        "trail": [0.0, 8.0, 12.0, 18.0],
    }

    def __init__(self, params=None):
        super().__init__(params)
        self.stoploss_pct = float(self.p["sl"])
        if float(self.p["trail"]) > 0:
            self.trailing_stop_pct = float(self.p["trail"])

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c, b, sb = int(self.p["conv"]), int(self.p["base"]), int(self.p["span_b"])
        src = heikin_ashi(df) if self.p["use_ha"] else df
        hi, lo, cl = src["high"], src["low"], src["close"]
        tenkan = (rolling_max(hi, c) + rolling_min(lo, c)) / 2
        kijun = (rolling_max(hi, b) + rolling_min(lo, b)) / 2
        df["ich_tenkan"] = tenkan
        df["ich_kijun"] = kijun
        df["ich_span_a"] = ((tenkan + kijun) / 2).shift(b)
        df["ich_span_b"] = ((rolling_max(hi, sb) + rolling_min(lo, sb)) / 2).shift(b)
        df["ich_close"] = cl
        df["ich_cloud_top"] = df[["ich_span_a", "ich_span_b"]].max(axis=1)
        df["ich_cloud_bot"] = df[["ich_span_a", "ich_span_b"]].min(axis=1)
        df["ich_chikou_ok"] = cl > cl.shift(b)      # price above where it was `base` bars ago
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        setup = (
            (df["ich_close"] > df["ich_cloud_top"])          # price clear of the whole cloud
            & (df["ich_span_a"] > df["ich_span_b"])          # cloud ahead is green
            & (df["ich_tenkan"] > df["ich_kijun"])           # bullish TK
            & df["ich_chikou_ok"]
            & (df["volume"] > 0)
        )
        return setup & ~setup.shift(1).fillna(False)         # fire once, on the rising edge

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        rule = self.p.get("exit_rule", "tk_cross")
        if rule == "kijun_break":
            return df["ich_close"] < df["ich_kijun"]
        if rule == "cloud_loss":                     # slowest: hold until price loses the whole cloud
            return df["ich_close"] < df["ich_cloud_bot"]
        return crossed_below(df["ich_tenkan"], df["ich_kijun"])   # classic TK death cross
