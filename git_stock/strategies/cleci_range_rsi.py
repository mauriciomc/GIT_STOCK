"""CleciRangeRSI — CleciRange plus an RSI confirmation filter on both sides.

Buy:  price crosses below entry_threshold (bottom buy_pct%% of the trailing range)
      AND RSI(14) > rsi_buy_min  — wait for RSI to show the bounce is underway,
      not still falling, rather than buying the exact bottom of a freefall.
Sell: price crosses above exit_threshold (top (100-sell_pct)%% of the range)
      AND RSI(14) < rsi_sell_max — wait for momentum to be cooling off, not still
      accelerating, rather than selling into a still-strengthening rally.

Same range-percentage band fix as CleciRange (see that file's docstring for why
the plain-multiplier version can invert its own buy/sell thresholds); this adds
the RSI gate as a genuinely separate, tunable filter on top - rsi_buy_min=0 and
rsi_sell_max=100 recover CleciRange exactly, so hyperopt can find "RSI doesn't
help" on its own if that's the honest answer, rather than the filter being
forced on.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import crossed_above, crossed_below, rolling_max, rolling_min, rsi
from ..strategy import Strategy


class CleciRangeRSI(Strategy):
    timeframe = "1d"
    space = {
        "window": (20, 400, 200, 5),
        "buy_pct": (5.0, 40.0, 16.0, 0.5),
        "sell_pct": (60.0, 95.0, 77.0, 0.5),
        "rsi_period": (7, 21, 14, 1),
        # 0/100 bounds recover CleciRange exactly - filter is optional, not forced
        "rsi_buy_min": (0.0, 50.0, 30.0, 0.5),
        "rsi_sell_max": (50.0, 100.0, 70.0, 0.5),
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
        df["rsi"] = rsi(df["close"], int(self.p["rsi_period"]))
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        crossed = crossed_below(df["close"], df["entry_threshold"])
        return crossed & (df["rsi"] > self.p["rsi_buy_min"])

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        if self.p["exit_mode"] in ("roi", "trail"):
            return pd.Series(False, index=df.index)
        crossed = crossed_above(df["close"], df["exit_threshold"])
        return crossed & (df["rsi"] < self.p["rsi_sell_max"])

    def score(self, df: pd.DataFrame) -> pd.Series:
        return df["exit_threshold"] / df["close"] - 1.0
