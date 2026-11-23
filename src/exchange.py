import os
import time
import datetime as dt

import pandas as pd
import yfinance as yf

# CSV schema the rest of the codebase (trader.prepare_data) expects:
#   index name "Datetime" (string "%Y-%m-%d %H:%M"), columns:
#   Open, High, Low, Close, Adj Close, Volume
_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

_TF = {
    "1m":  (7,   1,    "1m"),
    "5m":  (60,  5,    "5m"),
    "15m": (60,  15,   "15m"),
    "30m": (60,  30,   "30m"),
    "1h":  (180, 60,   "60m"),
}


def _download(ticker, period, interval, retries=3):
    """Modern-yfinance download normalised to a flat, Datetime-indexed frame."""
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker, period=period, interval=interval,
                auto_adjust=False, progress=False, threads=False,
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
            continue
        if df is None or df.empty:
            last_err = "empty response"
            time.sleep(1.0 * (attempt + 1))
            continue
        # yfinance >= 0.2.x returns a column MultiIndex (field, ticker); flatten it
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename_axis("Datetime", axis=0)
        # keep only the expected columns, in a stable order
        keep = [c for c in _COLS if c in df.columns]
        df = df[keep]
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
            df = df[[c for c in _COLS if c in df.columns]]
        df = df.dropna(how="all")
        if not df.empty:
            return df
    print(f"  {ticker}: download failed ({last_err})")
    return None


def get_data_from_yahoo(tickers="", daterange=2000, timeframe="1h"):
    if not os.path.exists("stock_data"):
        os.makedirs("stock_data")

    if timeframe in _TF:
        daterange, delta, interval = _TF[timeframe]
    else:
        delta, interval = 1440, "1d"

    now = dt.datetime.now()
    end = now.strftime("%Y-%m-%d %H:%M")
    period = f"{daterange}d"

    for ticker in tickers:
        path = f"stock_data/{ticker}_{timeframe}.csv"
        try:
            if not os.path.exists(path):
                print(f"Downloading {ticker}: period={period} interval={timeframe}")
                df = _download(ticker, period, interval)
                if df is None:
                    continue
                out = df.copy()
                out.index = out.index.strftime("%Y-%m-%d %H:%M")
                out.to_csv(path)
                continue

            # incremental update of an existing file
            try:
                df = pd.read_csv(path, parse_dates=["Datetime"]).set_index("Datetime")
            except Exception as e:  # noqa: BLE001
                print(f"Unable to read {path}: {e}")
                continue
            if df.empty:
                df = _download(ticker, period, interval)
                if df is None:
                    continue
                df.index = df.index.strftime("%Y-%m-%d %H:%M")
                df.to_csv(path)
                continue

            last = df.index[-1].to_pydatetime()
            next_bar = (last + dt.timedelta(minutes=delta)).strftime("%Y-%m-%d %H:%M")
            print(f"ticker={ticker} | last={df.index[-1]:%Y-%m-%d %H:%M} | now={end}")
            if next_bar >= end:
                continue

            fresh = _download(ticker, period, interval)
            if fresh is None:
                continue
            merged = pd.concat([df, fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged.index = merged.index.strftime("%Y-%m-%d %H:%M")
            merged.to_csv(path)

        except Exception as e:  # noqa: BLE001
            print(f"Exception for {ticker}: {e}")
            continue
