"""SwingTrend - the realistic version of "buy every valley, sell every peak".

You can't know a valley is a valley until price has already bounced off it, so
entry requires a CONFIRMED reversal: price closes reversal_pct% above its own
trailing lookback-day low. Exit is a trailing stop off the position's own
high-water mark - lets winners run instead of capping them at a fixed target,
the polar opposite of BbRsi's mean-reversion design (fixed/generous take-profit,
sell near a Bollinger high).

Purpose-built to catch relentless trend stocks like AMD/NVDA, which BbRsi
structurally can't (confirmed: zero trades on both across the full 2015-2026
backtest) because they rarely give a genuine "oversold" reading.

No RSI, no Bollinger, no sector weighting yet - deliberately the simplest
version: one entry rule (breakout off a trailing low) and the engine's existing
trailing-stop machinery for the exit, plus a wide hard stop as a catastrophe
backstop, not the primary exit.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import crossed_above, rolling_min
from ..strategy import Strategy


class SwingTrend(Strategy):
    timeframe = "1d"
    space = {
        "lookback": (10, 100, 20, 5),
        "reversal_pct": (3.0, 30.0, 10.0, 1.0),
        "trail_pct": (5.0, 30.0, 15.0, 1.0),
        "sl": (10.0, 40.0, 25.0, 1.0),
    }

    def __init__(self, params=None):
        super().__init__(params)
        self.trailing_stop_pct = float(self.p["trail_pct"])
        self.stoploss_pct = float(self.p["sl"])

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        w = int(self.p["lookback"])
        df["swing_low"] = rolling_min(df["low"], w)
        df["entry_threshold"] = df["swing_low"] * (1 + self.p["reversal_pct"] / 100.0)
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return crossed_above(df["close"], df["entry_threshold"])

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)  # the trailing stop is the only exit
