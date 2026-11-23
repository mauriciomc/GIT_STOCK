"""BB + RSI + TEMA mean-reversion — modernised from the recovered ``strategies/bbrsi.py``."""
from __future__ import annotations

import pandas as pd

from ..indicators import bbands, crossed_above, crossed_below, ema, rsi
from ..strategy import Strategy


class BbRsi(Strategy):
    timeframe = "1d"
    space = {
        "bb_window": (10, 40, 20, 2),
        "rsi_window": (7, 30, 14, 1),
        "rsi_buy": (20, 45, 30, 1),
        "rsi_sell": (55, 80, 65, 1),
    }

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        w = int(self.p["bb_window"])
        df["bb_low"], df["bb_mid"], df["bb_high"] = bbands(df["close"], w, 2.0)
        df["tema"] = ema(ema(ema(df["close"], 14), 14), 14) * 3 - ema(ema(df["close"], 14), 14) * 3 + ema(df["close"], 14)
        df["rsi"] = rsi(df["close"], int(self.p["rsi_window"]))
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return (
            (df["close"] > df["bb_low"])
            & (df["tema"] > df["tema"].shift(1))
            & crossed_above(df["rsi"], float(self.p["rsi_buy"]))
        )

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return (
            (df["close"] > df["bb_high"]) | crossed_below(df["close"], df["bb_mid"])
        ) & crossed_below(df["rsi"], float(self.p["rsi_sell"]))
