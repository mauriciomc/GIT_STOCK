"""Vectorised indicator helpers — no TA-Lib / qtpylib dependency.

The old strategies relied on qtpylib's monkey-patched ``Series.crossed_above`` and
TA-Lib. Both are replaced here with plain pandas/numpy so the package runs on a
clean modern Python.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def crossed_above(a: pd.Series, b: pd.Series | float) -> pd.Series:
    b = pd.Series(b, index=a.index) if np.isscalar(b) else b
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossed_below(a: pd.Series, b: pd.Series | float) -> pd.Series:
    b = pd.Series(b, index=a.index) if np.isscalar(b) else b
    return (a < b) & (a.shift(1) >= b.shift(1))


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = roll_up / roll_down
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((roll_down == 0) & (roll_up > 0), 100.0)   # only gains -> 100
    out = out.mask((roll_down == 0) & (roll_up == 0), 50.0)   # flat -> neutral
    return out


def bbands(s: pd.Series, window: int = 20, stds: float = 2.0):
    mid = sma(s, window)
    sd = s.rolling(window, min_periods=window).std()
    return mid - stds * sd, mid, mid + stds * sd


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def linreg_angle(s: pd.Series, window: int = 2) -> pd.Series:
    """Angle (degrees) of a rolling least-squares fit — TA-Lib LINEARREG_ANGLE.
    Vectorised: for equally-spaced x the OLS slope is a fixed linear filter on y."""
    w = int(window)
    if w <= 1:
        return pd.Series(0.0, index=s.index)
    if w == 2:
        return np.degrees(np.arctan(s.diff()))
    x = np.arange(w) - (w - 1) / 2.0
    coef = x / (x ** 2).sum()                       # slope = sum(coef[k] * y_window[k])
    y = s.to_numpy(float)
    slope = np.full(len(y), np.nan)
    valid = np.convolve(y, coef[::-1], mode="valid")
    slope[w - 1:] = valid
    return pd.Series(np.degrees(np.arctan(slope)), index=s.index)


def rolling_min(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).min()


def rolling_max(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).max()


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    ha_close = (o + h + lo + c) / 4.0
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_close)):          # recursive; numpy loop, ~50x faster than .iloc
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([lo, ha_open, ha_close])
    return pd.DataFrame({"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close},
                        index=df.index)
