"""Fetch full available history (back to each ticker's IPO where yfinance has it) into a
SEPARATE cache dir, so this never touches `stock_data/` while other backtests/hyperopts
are reading it concurrently.

    ./.venv/bin/python scripts/fetch_full_history.py
"""
from __future__ import annotations

from git_stock.config import Config
from git_stock.data import DataFeed

CACHE = "stock_data_full"


def main() -> None:
    cfg = Config.load("config/config.json")
    feed = DataFeed(CACHE, cfg.timeframe)
    print(f"fetching {len(cfg.tickers)} tickers into {CACHE}/ (period=12000d, ~33y back)…")
    feed.update(cfg.tickers, daterange=12000)
    data = feed.load_many(cfg.tickers)
    covers_2000_2006 = sorted(sym for sym, df in data.items() if df.index[0] <= __import__("pandas").Timestamp("2000-06-01"))
    print(f"\n{len(data)}/{len(cfg.tickers)} tickers fetched with any data")
    print(f"{len(covers_2000_2006)} tickers have data starting on/before 2000-06-01 "
          f"(usable for a 2000-2006 backtest):")
    print(" ".join(covers_2000_2006))


if __name__ == "__main__":
    main()
