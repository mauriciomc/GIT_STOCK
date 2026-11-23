"""Market data via yfinance, with an on-disk CSV cache.

Replaces the old ``src/exchange.py`` which was pinned to yfinance 0.1.64 and stopped
working when Yahoo retired that endpoint (2024-08). Modern yfinance returns a column
MultiIndex and defaults ``auto_adjust=True`` — both handled here.
"""
from __future__ import annotations

import logging
import pathlib
import time

import pandas as pd

log = logging.getLogger("git_stock.data")

OHLCV = ["open", "high", "low", "close", "adj_close", "volume"]
_YF_MAP = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
           "Adj Close": "adj_close", "Volume": "volume"}
# every spelling of each field we might meet in a cached CSV -> canonical name
_ALIASES = {
    "open": "open", "Open": "open",
    "high": "high", "High": "high",
    "low": "low", "Low": "low",
    "close": "close", "Close": "close",
    "adj_close": "adj_close", "Adj Close": "adj_close", "adj close": "adj_close",
    "adjclose": "adj_close", "Adj_Close": "adj_close",
    "volume": "volume", "Volume": "volume",
}


def _repair_bad_wicks(out: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Fix single-bar low/high prints that are wildly inconsistent with that SAME
    bar's own open and close - seen in the wild as a ~100x GBX-pence/GBP-pounds unit
    glitch on LSE (.L) tickers (e.g. BP.L 2019-12-13: open/high/close all ~460-473,
    low printed as 4.686), confirmed reproducible straight from a fresh yfinance
    re-download (a Yahoo data-quality issue, not anything introduced by this cache).
    A real crash still drags the close down; a bar where open AND close both look
    normal but low or high doesn't is a bad print, not a real trade. Clamped to the
    open/close bound instead of guessing a "true" price.

    Deliberately conservative: only clamps when low/high is beyond 2x the open-close
    range, since no genuine single-bar move looks like that. A looser, "did the move
    revert within N bars" heuristic was tried and rejected - it false-positived on
    real violent days (e.g. OXY's 2020-03-09 oil-price-war crash), because during an
    already-strong trend a modest same-day wick can look "unsustained" relative to
    the next few bars even though nothing was wrong with the print. Don't loosen this
    without a materially better test; guessing wrong here corrupts real data, which
    is worse than leaving a few genuine bad ticks unrepaired."""
    if out.empty:
        return out
    lo_bound = out[["open", "close"]].min(axis=1)
    hi_bound = out[["open", "close"]].max(axis=1)
    bad_low = out["low"] < lo_bound * 0.5
    bad_high = out["high"] > hi_bound * 1.5
    n_bad = int(bad_low.sum() + bad_high.sum())
    if n_bad:
        log.warning("%s: repairing %d bad OHLC wick(s) (open/close normal, "
                    "low/high off by >2x that same bar)", ticker or "?", n_bad)
        out.loc[bad_low, "low"] = lo_bound[bad_low]
        out.loc[bad_high, "high"] = hi_bound[bad_high]
    return out


def _normalize(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame | None:
    """Collapse any mix of column spellings (Open/open, 'Adj Close'/adj_close, …) into
    canonical lowercase OHLCV, coalescing duplicate-case columns. Repairs the
    mixed-schema CSVs an earlier buggy merge produced, and any single-bar bad wicks."""
    if df is None or df.empty:
        return None
    out = pd.DataFrame(index=pd.to_datetime(df.index))
    for col in df.columns:
        canon = _ALIASES.get(str(col).strip())
        if not canon:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        out[canon] = s if canon not in out else out[canon].combine_first(s)
    if "close" not in out.columns:
        return None
    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]
    for c in ("open", "high", "low", "volume"):
        if c not in out.columns:
            out[c] = out["close"]
    out = out[list(OHLCV)].dropna(subset=["close"])
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return _repair_bad_wicks(out, ticker)
# yfinance interval per timeframe, plus the max lookback Yahoo allows for it
_INTERVAL = {"1m": ("1m", 7), "5m": ("5m", 60), "15m": ("15m", 60),
             "30m": ("30m", 60), "1h": ("60m", 730), "1d": ("1d", None)}


def _yf_download(ticker: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame | None:
    import yfinance as yf

    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=False, progress=False, threads=False)
        except Exception as e:  # noqa: BLE001
            log.warning("%s download error (try %d): %s", ticker, attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
            continue
        if df is None or df.empty:
            time.sleep(1.0 * (attempt + 1))
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=_YF_MAP)
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]
        df = df[[c for c in OHLCV if c in df.columns]].dropna(how="all")
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df[~df.index.duplicated(keep="last")].sort_index()
    log.warning("%s: no data after %d tries", ticker, retries)
    return None


class DataFeed:
    """Fetches + caches OHLCV. ``load()`` returns a clean float DataFrame indexed by date."""

    def __init__(self, cache_dir: str | pathlib.Path = "data_cache", timeframe: str = "1d"):
        self.dir = pathlib.Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timeframe = timeframe
        self.interval, _ = _INTERVAL.get(timeframe, ("1d", None))

    def _path(self, ticker: str) -> pathlib.Path:
        return self.dir / f"{ticker}_{self.timeframe}.csv"

    def _read_cached(self, p: pathlib.Path, ticker: str = "") -> pd.DataFrame | None:
        if not p.exists():
            return None
        try:
            return _normalize(pd.read_csv(p, index_col=0, parse_dates=True), ticker)
        except Exception as e:  # noqa: BLE001
            log.warning("could not read %s: %s", p.name, e)
            return None

    def update(self, tickers: list[str], daterange: int = 2000) -> None:
        period = f"{daterange}d"
        for t in tickers:
            fresh = _yf_download(t, period, self.interval)
            p = self._path(t)
            old = self._read_cached(p, t)
            if fresh is None:
                if old is not None:
                    old.to_csv(p)          # rewrite clean even if the fetch failed
                continue
            fresh = _repair_bad_wicks(fresh, t)
            if old is not None:
                fresh = pd.concat([old, fresh])
                fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
            fresh.to_csv(p)
            log.info("%s: %d rows -> %s", t, len(fresh), fresh.index[-1].date())

    def load(self, ticker: str) -> pd.DataFrame | None:
        return self._read_cached(self._path(ticker), ticker)

    def repair(self, tickers: list[str]) -> tuple[int, int]:
        """Rewrite every cached CSV in canonical form. Fixes mixed-schema files."""
        ok = bad = 0
        for t in tickers:
            p = self._path(t)
            if not p.exists():
                continue
            df = self._read_cached(p, t)
            if df is None or len(df) < 5:
                bad += 1
                continue
            df.to_csv(p)
            ok += 1
        return ok, bad

    def load_many(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        out = {}
        for t in tickers:
            df = self.load(t)
            if df is not None and len(df) > 5:
                out[t] = df
        return out
