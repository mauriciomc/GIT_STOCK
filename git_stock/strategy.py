"""Strategy base class.

A strategy declares a parameter ``space`` (used by both the default value and the
optimiser), then implements three vectorised hooks over an OHLCV DataFrame:

    populate_indicators(df) -> df          # add indicator columns
    entry_signal(df)        -> bool Series # True where a long is opened
    exit_signal(df)         -> bool Series # True where the long is closed
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# a param spec is a tuple (low, high, default[, step]) for numeric,
# or a list [choices...] for categorical.  Space = {name: spec}.
Space = dict


def default_params(space: Space) -> dict:
    out = {}
    for name, spec in space.items():
        out[name] = spec[2] if isinstance(spec, tuple) else spec[0]
    return out


def actionable_signals(entry: pd.Series, exit: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Collapse raw signal series into the ones a single-position long strategy would
    actually act on: an entry only fires when flat, an exit only when in a position.

    Mirrors the backtester's one-position-per-symbol rule (stops/TP aside), so charts and
    reports show the trades that were really taken instead of every indicator crossing.
    On a bar with both signals: enter if flat, exit if in a position.
    """
    e = entry.to_numpy(bool)
    x = exit.to_numpy(bool)
    act_e = np.zeros(len(e), bool)
    act_x = np.zeros(len(x), bool)
    in_pos = False
    for i in range(len(e)):
        if not in_pos and e[i]:
            act_e[i] = in_pos = True
        elif in_pos and x[i]:
            act_x[i] = True
            in_pos = False
    return (pd.Series(act_e, index=entry.index, name="entry"),
            pd.Series(act_x, index=exit.index, name="exit"))


class Strategy(ABC):
    timeframe: str = "1d"
    space: Space = {}
    # optional risk overrides (else the backtest config's values are used)
    stoploss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    trailing_activation_pct: float | None = None
    # "entry" (default) measures trailing_activation_pct off the position's own entry
    # price; "exit_threshold" measures it off the strategy's exit_threshold column at
    # each bar instead (needs populate_indicators to produce that column)
    trail_activation_from: str | None = None
    cooldown_bars: int | None = None

    def __init__(self, params: dict | None = None):
        self.p = {**default_params(self.space), **(params or {})}

    @property
    def name(self) -> str:
        return type(self).__name__.lower()

    def describe(self) -> str:
        """One-line 'name · k=v k=v' summary of the running configuration."""
        parts = [f"{k}={self.p.get(k)}" for k in self.space]
        return f"{self.name} · " + " ".join(parts) if parts else self.name

    @abstractmethod
    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def entry_signal(self, df: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def exit_signal(self, df: pd.DataFrame) -> pd.Series: ...

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Optional cross-sectional 'how much upside is left' estimate, comparable across
        symbols/time, used by the backtest's portfolio rotation (BacktestConfig.rotate):
        when at max_open_trades and a new entry fires, the weakest-scoring OPEN position is
        force-closed to free a slot for the new candidate if its score clears it by enough.
        Default: NaN everywhere == "no score available" == rotation disabled for this
        strategy (the engine falls back to just dropping the signal, today's behaviour)."""
        return pd.Series(np.nan, index=df.index)

    # convenience used by backtest/signals
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.populate_indicators(df.copy())
        df["entry"] = self.entry_signal(df).fillna(False).astype(bool)
        df["exit"] = self.exit_signal(df).fillna(False).astype(bool)
        df["score"] = self.score(df)
        return df
