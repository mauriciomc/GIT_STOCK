"""CleciRangeCross - simplest possible reclaim/breakout variant of CleciRangeRSI.

Same band construction as CleciRangeRSI/CleciRange, untouched (rolling window,
buy_pct/sell_pct of the SAME range - entry_threshold=green line, exit_threshold=
red line, correct ordering guaranteed by construction). Only the entry/exit
RULES change:

Buy: price crosses ABOVE entry_threshold (the green line) OR crosses ABOVE
exit_threshold (the red line) - a plain close-to-close crossover, no same-bar
wick/bounce pattern required (that's CleciRangeBounce; this is the simpler
cousin). AND RSI(period) > rsi_buy_min (optional gate, 0 recovers "no filter").

Sell: price crosses BELOW exit_threshold (the red line) AND RSI(period) <
rsi_sell_max.

No take-profit, no stoploss, no trailing stop of any kind - the crossed-below-
red signal above is the ONLY exit. stoploss_pct/take_profit_pct are pinned wide
open so BacktestConfig's own defaults (a 5% stop otherwise applies to every
strategy that doesn't override it) can't silently interfere.

Regime gate: buys are additionally blocked unless BOTH lines are rising OR
FLAT-COMING-OFF-A-RISE over the last slope_window candles - i.e. don't buy a
reclaim/breakout while the whole trailing range is still sliding downward, but
DO allow a sideways consolidation as long as it followed a genuine longer-run
(long_slope_window) uptrend rather than being noise inside a downtrend or a
truly flat market. Concretely, per line:

    rising_or_pausing = slope > 0  OR  (slope == 0  AND  long_slope > 0)

Motivated directly by the MGLU3.SA trade (-66.7%, held 2023-08 to 2025-12): the
entry was taken during a reclaim bounce while both lines were declining in
lockstep with the price the entire time. The flat/0 case matters because
rolling-min/max lines are step functions - they sit PERFECTLY flat for real
stretches whenever the window's own extreme hasn't been touched, which is a
meaningful "still holding a prior high/low, not fresh but not failing either"
state, not just floating-point noise - so it's checked as literal equality.
"Slope" is the plain difference over N candles, not a full regression - only
the sign matters here, and sign(difference) == sign(angle) always.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import crossed_above, crossed_below, rolling_max, rolling_min, rsi
from ..strategy import Strategy


class CleciRangeCross(Strategy):
    timeframe = "1d"
    # no exit is silently applied by the engine unless explicitly pinned open
    stoploss_pct = 99.0
    take_profit_pct = 10000.0
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
        "cooldown_bars": (0, 20, 0, 1),
        "slope_window": (5, 40, 15, 1),
        "long_slope_window": (60, 250, 120, 5),
    }

    def __init__(self, params=None):
        super().__init__(params)
        if self.p["buy_pct"] >= self.p["sell_pct"]:
            raise ValueError(
                f"buy_pct ({self.p['buy_pct']}) must be < sell_pct ({self.p['sell_pct']}) "
                "or the bands invert by construction"
            )
        self.cooldown_bars = int(self.p["cooldown_bars"])

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        w = int(self.p["window"])
        hi_src = df["high"] if self.p.get("high_band", "low") == "high" else df["low"]
        df["band_low"] = rolling_min(df["low"], w)
        df["band_high"] = rolling_max(hi_src, w)

        rng = df["band_high"] - df["band_low"]
        df["entry_threshold"] = df["band_low"] + rng * (self.p["buy_pct"] / 100.0)
        df["exit_threshold"] = df["band_low"] + rng * (self.p["sell_pct"] / 100.0)
        df["rsi"] = rsi(df["close"], int(self.p["rsi_period"]))

        sw = int(self.p["slope_window"])
        lsw = int(self.p["long_slope_window"])
        df["green_slope"] = df["entry_threshold"] - df["entry_threshold"].shift(sw)
        df["red_slope"] = df["exit_threshold"] - df["exit_threshold"].shift(sw)
        df["green_slope_long"] = df["entry_threshold"] - df["entry_threshold"].shift(lsw)
        df["red_slope_long"] = df["exit_threshold"] - df["exit_threshold"].shift(lsw)
        return df

    @staticmethod
    def _rising_or_pausing(slope: pd.Series, slope_long: pd.Series) -> pd.Series:
        return (slope > 0) | ((slope == 0) & (slope_long > 0))

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        green = crossed_above(df["close"], df["entry_threshold"])
        red = crossed_above(df["close"], df["exit_threshold"])
        green_ok = self._rising_or_pausing(df["green_slope"], df["green_slope_long"])
        red_ok = self._rising_or_pausing(df["red_slope"], df["red_slope_long"])
        return (green | red) & (df["rsi"] > self.p["rsi_buy_min"]) & green_ok & red_ok

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        crossed = crossed_below(df["close"], df["exit_threshold"])
        return crossed & (df["rsi"] < self.p["rsi_sell_max"])

    def score(self, df: pd.DataFrame) -> pd.Series:
        return df["exit_threshold"] / df["close"] - 1.0
