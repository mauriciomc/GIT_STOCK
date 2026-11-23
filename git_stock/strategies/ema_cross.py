"""EMA-cross trend follower — the pre-class ``strategy_ema_cross`` from the old trader.py,
one of the candidates for the 2024 ~47 % run.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import crossed_above, crossed_below, ema, linreg_angle
from ..strategy import Strategy


class EmaCross(Strategy):
    timeframe = "1d"
    space = {
        "fast": (5, 20, 12, 1),
        "mid": (20, 80, 50, 5),
        "slow": (100, 250, 200, 10),
        "min_angle": (0.0, 1.0, 0.0, 0.05),
    }

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ema_fast"] = ema(df["close"], int(self.p["fast"]))
        df["ema_mid"] = ema(df["close"], int(self.p["mid"]))
        df["ema_slow"] = ema(df["close"], int(self.p["slow"]))
        df["angle_fast"] = linreg_angle(df["ema_fast"], 2)
        df["angle_slow"] = linreg_angle(df["ema_slow"], 2)
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        a = self.p["min_angle"]
        return (
            crossed_above(df["close"], df["ema_fast"])
            & (df["ema_fast"] > df["ema_mid"])
            & (df["ema_mid"] > df["ema_slow"])
            & (df["angle_fast"] > a)
            & (df["angle_slow"] > a)
            & (df["volume"] > 0)
        )

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return crossed_below(df["close"], df["ema_fast"]) & (df["volume"] > 0)
