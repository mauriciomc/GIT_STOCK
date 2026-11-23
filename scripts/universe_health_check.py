"""Lightweight, periodic universe-maintenance check (quarterly cadence suggested).

This is deliberately cheap and mechanical - NOT a re-optimization. It flags:
  - stale tickers: no new data in a while relative to the rest of the universe
    (the usual signature of a delisting, ticker-symbol change, or a broken feed -
    we already found 12/207 like this, frozen at 2024-08-01, during this study)
  - short-history tickers: too little data for stage-1 isolated hyperopt to mean
    much for that sector yet (recent IPOs)
  - sectors left with very few tickers after the above, which concentrates
    capital onto a handful of names more than the sector weight search assumed

It does not remove or modify anything - it only reports, for a human to act on.

    ./.venv/bin/python scripts/universe_health_check.py [--stale-days 10]
"""
from __future__ import annotations

import argparse

import pandas as pd

from git_stock.config import Config
from git_stock.data import DataFeed

MIN_SECTOR_TICKERS = 5     # below this, a sector's weight is riding on very few names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale-days", type=int, default=10,
                    help="flag a ticker whose last bar is this many calendar days "
                         "or more behind the universe's freshest bar")
    ap.add_argument("--min-history-days", type=int, default=365,
                    help="flag a ticker with less than this many days of history")
    ap.add_argument("--cache", default="stock_data")
    ap.add_argument("--config", default="config/config.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data = DataFeed(args.cache, cfg.timeframe).load_many(cfg.tickers)
    sector_of = {t: sec for sec, ts in cfg.sectors.items() for t in ts}

    freshest = max(df.index[-1] for df in data.values())
    print(f"universe: {len(cfg.tickers)} configured, {len(data)} with cached data, "
          f"freshest bar {freshest.date()}")

    stale, short_history, missing = [], [], []
    for t in cfg.tickers:
        df = data.get(t)
        if df is None or df.empty:
            missing.append(t)
            continue
        age_days = (freshest - df.index[-1]).days
        if age_days >= args.stale_days:
            stale.append((t, sector_of.get(t, "?"), df.index[-1].date(), age_days))
        span_days = (df.index[-1] - df.index[0]).days
        if span_days < args.min_history_days:
            short_history.append((t, sector_of.get(t, "?"), df.index[0].date(), span_days))

    print(f"\n--- missing from cache entirely ({len(missing)}) ---")
    for t in missing:
        print(f"  {t:10} sector={sector_of.get(t, '?')}")

    print(f"\n--- stale (last bar >= {args.stale_days}d behind {freshest.date()}) ({len(stale)}) ---")
    for t, sec, last, age in sorted(stale, key=lambda r: -r[3]):
        print(f"  {t:10} sector={sec:14} last_bar={last}  {age}d stale")

    print(f"\n--- short history (< {args.min_history_days}d span) ({len(short_history)}) ---")
    for t, sec, first, span in sorted(short_history, key=lambda r: r[3]):
        print(f"  {t:10} sector={sec:14} first_bar={first}  {span}d of history")

    print(f"\n--- sector ticker counts after excluding missing/stale ---")
    bad = {t for t, *_ in stale} | set(missing)
    for sec in sorted(cfg.sectors):
        total = len(cfg.sectors[sec])
        usable = sum(1 for t in cfg.sectors[sec] if t not in bad)
        flag = "  <-- below floor" if usable < MIN_SECTOR_TICKERS else ""
        print(f"  {sec:14} {usable}/{total} usable{flag}")

    if not stale and not missing and not short_history:
        print("\nno issues found.")


if __name__ == "__main__":
    main()
