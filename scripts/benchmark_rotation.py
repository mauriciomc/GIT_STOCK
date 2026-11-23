"""Capital-matched benchmarks for the OOS window — no infinite-money buy & hold.

The full-universe (207-ticker) equal-weight buy & hold used earlier isn't a fair
comparison: it assumes capital to hold every name at once. These two benchmarks respect
the same ~40-position constraint (`max_open_trades`) the strategies actually trade under:

  A. Random-40 buy & hold, no rebalancing — N draws of 40 tickers held the whole OOS
     window, distribution reported. Isolates "being diversified across ~40 names"
     from any skill.
  B. Monthly top-40 momentum rotation — naive "keep rotating into whatever looks best"
     with zero real signal (pure trailing-return momentum), rebalanced monthly, same
     realistic per-side cost (0.4% = fee+slippage+half-spread) as the strategy bake-off's
     dirty cost model. This is the actual baseline for "I have to keep finding better
     positions" — does blind rotation alone beat the signal-driven strategies?

    ./.venv/bin/python scripts/benchmark_rotation.py
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from git_stock.config import Config
from git_stock.data import DataFeed

OOS_START, OOS_END = "2022-07-01", "2030-01-01"
N_POS = 40                 # matches config.json's max_open_trades
LOOKBACK_DAYS = 126        # ~6 trading months, momentum ranking window
COST_PER_SIDE = 0.003 + 0.0005 + 0.0010 / 2   # fee + slippage + half-spread = 0.4%, matches BacktestConfig "dirty"
RF = 0.02


def _equity_stats(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    years = len(rets) / 252
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vol = rets.std(ddof=1) * np.sqrt(252)
    down = rets[rets < 0]
    dvol = down.std(ddof=1) * np.sqrt(252) if len(down) > 1 else 0.0
    peak = equity.cummax()
    mdd = float((equity / peak - 1.0).min())
    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round((cagr - RF) / vol, 2) if vol else None,
        "sortino": round((cagr - RF) / dvol, 2) if dvol else None,
        "max_drawdown_pct": round(mdd * 100, 2),
    }


def main() -> None:
    cfg = Config.load("config/config.json")
    data = DataFeed("stock_data", cfg.timeframe).load_many(cfg.tickers)
    print(f"{len(data)} tickers loaded")

    panel = pd.DataFrame({sym: df["close"] for sym, df in data.items()})
    panel = panel.sort_index()
    oos = panel.loc[(panel.index >= OOS_START) & (panel.index <= OOS_END)]
    oos = oos.dropna(axis=1, how="all")
    print(f"OOS window: {oos.index[0].date()} .. {oos.index[-1].date()}  ({len(oos)} bars, "
          f"{oos.shape[1]} tickers with any data)\n")

    # ---------------------------------------------------------------
    # A. Random-40 buy & hold, no rebalance
    # ---------------------------------------------------------------
    start_date = oos.index[0]
    eligible = oos.columns[oos.loc[start_date].notna()].tolist()
    print(f"=== A. Random-{N_POS} buy & hold (no rebalance), {len(eligible)} eligible at start ===")
    rng = random.Random(7)
    draws, paths = [], []
    for _ in range(300):
        basket = rng.sample(eligible, min(N_POS, len(eligible)))
        sub = oos[basket].ffill()
        # each name entered at start, one-side cost only (no exit until the end, so we
        # only charge the entry — a genuinely "buy & hold" cost profile)
        rel = sub / sub.iloc[0] * (1 - COST_PER_SIDE)
        eq = rel.mean(axis=1)  # equal-weight
        draws.append(_equity_stats(eq)["total_return_pct"])
        paths.append(eq)
    draws = pd.Series(draws)
    print(f"  {len(draws)} draws: mean {draws.mean():.1f}%  median {draws.median():.1f}%  "
          f"p10 {draws.quantile(.1):.1f}%  p90 {draws.quantile(.9):.1f}%")
    # per-draw risk stats, averaged — each draw's OWN Sharpe/maxDD, not an ensemble-smoothed curve
    risk = pd.DataFrame([_equity_stats(p) for p in paths])
    print(f"  avg per-draw: Sharpe {risk['sharpe'].mean():.2f}  Sortino {risk['sortino'].mean():.2f}  "
          f"maxDD {risk['max_drawdown_pct'].mean():.1f}%  (worst-draw maxDD {risk['max_drawdown_pct'].min():.1f}%)\n")

    # ---------------------------------------------------------------
    # B. Monthly top-40 momentum rotation, realistic costs
    # ---------------------------------------------------------------
    print(f"=== B. Monthly top-{N_POS} momentum rotation (naive, {LOOKBACK_DAYS}d lookback), "
          f"{COST_PER_SIDE:.2%}/side cost ===")
    month_ends = oos.resample("ME").last().index
    rebal_dates = [panel.index[panel.index.get_indexer([d], method="pad")[0]] for d in month_ends]
    rebal_dates = sorted(set(d for d in rebal_dates if d >= start_date))

    equity = [1.0]
    dates = [start_date]
    held: dict[str, float] = {}  # symbol -> weight

    def portfolio_value_path(d0, d1, weights):
        seg = panel.loc[(panel.index > d0) & (panel.index <= d1)]
        if seg.empty or not weights:
            return [], []
        syms = list(weights)
        base = panel.loc[d0, syms]
        rel = seg[syms].ffill() / base
        rel = rel.ffill().fillna(1.0)
        w = pd.Series(weights)
        port = (rel * w).sum(axis=1)
        return list(seg.index), list(port.values)

    cursor = start_date
    cum = 1.0
    for i, rd in enumerate(rebal_dates):
        # momentum rank using data up to and including `cursor` (no lookahead past it)
        hist = panel.loc[:cursor]
        if len(hist) <= LOOKBACK_DAYS:
            continue
        px_now, px_then = hist.iloc[-1], hist.iloc[-LOOKBACK_DAYS]
        mom = (px_now / px_then - 1.0).dropna()
        mom = mom[hist.iloc[-1].notna()]
        top = mom.sort_values(ascending=False).head(N_POS).index.tolist()
        new_w = {s: 1.0 / len(top) for s in top} if top else {}

        turnover = set(held) ^ set(new_w)  # symmetric diff = names added or dropped
        cost = len(turnover) * (1.0 / max(len(new_w), 1)) * COST_PER_SIDE
        cum *= (1 - cost)

        end = rd if i < len(rebal_dates) else oos.index[-1]
        ds, vals = portfolio_value_path(cursor, end, new_w)
        if vals:
            scale = cum / vals[0] if vals[0] else cum
            for d, v in zip(ds, vals):
                dates.append(d)
                equity.append(v * scale)
            cum = equity[-1]
        held = new_w
        cursor = end

    # tail segment to the last OOS bar
    if cursor < oos.index[-1] and held:
        ds, vals = portfolio_value_path(cursor, oos.index[-1], held)
        if vals:
            scale = cum / vals[0] if vals[0] else cum
            for d, v in zip(ds, vals):
                dates.append(d)
                equity.append(v * scale)

    eq = pd.Series(equity, index=pd.Index(dates)).groupby(level=0).last().sort_index()
    st = _equity_stats(eq)
    print(f"  {len(rebal_dates)} monthly rebalances, {eq.index[0].date()}..{eq.index[-1].date()}")
    print(f"  return {st['total_return_pct']}%  CAGR {st['cagr_pct']}%  Sharpe {st['sharpe']}  "
          f"Sortino {st['sortino']}  maxDD {st['max_drawdown_pct']}%")


if __name__ == "__main__":
    main()
