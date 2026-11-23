"""Take the top 5% of IS trials from each of the last hyperopt runs (clecirange,
clecirangersi, clecirangersiportfolio) and re-backtest every one of them over the
FULL available history (not just the 4.3y OOS window), to find the single most
robust config across the longest period we actually have data for.

Broader than just re-checking the top-8-by-IS-score (the main hyperopt scripts'
own selection): a wider net over more trials, evaluated over more years, catches
configs that are consistently good rather than ones that just top the IS ranking
or got lucky in the shorter OOS window.

    ./.venv/bin/python scripts/longperiod_bestpct.py
"""
from __future__ import annotations

import dataclasses
import math

import pandas as pd

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.optimize import PORTFOLIO_SPACE
from git_stock.strategies.cleci_range import CleciRange
from git_stock.strategies.cleci_range_rsi import CleciRangeRSI

LONG_START, LONG_END = "2015-12-22", "2030-01-01"   # full available history
TOP_PCT = 0.05

JOBS = [
    ("clecirange", "bt/is_table_clecirange.csv", CleciRange, False),
    ("clecirangersi", "bt/is_table_clecirangersi.csv", CleciRangeRSI, False),
    ("clecirangersiportfolio", "bt/is_table_clecirangersiportfolio.csv", CleciRangeRSI, True),
]


def _row(tag, st):
    return {
        f"{tag}_return%": st["total_return_pct"], f"{tag}_CAGR%": st["cagr_pct"],
        f"{tag}_Sharpe": st["sharpe"], f"{tag}_Sortino": st["sortino"],
        f"{tag}_maxDD%": st["max_drawdown_pct"], f"{tag}_PF": st["profit_factor"],
        f"{tag}_win%": st["win_rate_pct"], f"{tag}_trades": st["trades"],
        f"{tag}_hold": st["avg_bars_held"], f"{tag}_realised%": st["realised_pct"],
        f"{tag}_unrealised%": st["unrealised_pct"], f"{tag}_openAtEnd": st["open_at_end"],
    }


def main() -> None:
    cfg = Config.load("config/config.json")
    data = DataFeed("stock_data", cfg.timeframe).load_many(cfg.tickers)
    print(f"{len(data)} tickers loaded  ·  long period {LONG_START}..{LONG_END}\n")

    base_bt = BacktestConfig(
        initial_balance=cfg.initial_balance, stake_amount=cfg.stake_amount,
        max_open_trades=cfg.max_open_trades, fee=0.003, slippage_pct=0.0005,
        spread_pct=0.0010, fill_next_open=True, gap_stops=True, timeframe=cfg.timeframe)

    all_rows = []
    for name, path, cls, is_portfolio in JOBS:
        try:
            table = pd.read_csv(path)
        except FileNotFoundError:
            print(f"  ({path} not found, skipping {name})")
            continue
        table = table.sort_values("score", ascending=False).reset_index(drop=True)
        n_top = max(1, math.ceil(len(table) * TOP_PCT))
        top = table.head(n_top)
        print(f"=== {name}: top {n_top} of {len(table)} trials ({TOP_PCT:.0%}), "
              f"IS score range {top['score'].min():.3f}..{top['score'].max():.3f} ===", flush=True)

        strat_names = list(cls.space)
        bt_names = list(PORTFOLIO_SPACE) if is_portfolio else []
        for i, (_, r) in enumerate(top.iterrows()):
            strat_params = {k: r[k] for k in strat_names}
            try:
                strat = cls(strat_params)
            except Exception:
                continue
            trial_cfg = dataclasses.replace(base_bt, **{k: r[k] for k in bt_names}) if is_portfolio else base_bt
            try:
                st = run(data, strat, trial_cfg, LONG_START, LONG_END).stats
            except Exception:
                continue
            if st["trades"] < 20:
                continue
            full_params = {**strat_params, **({k: r[k] for k in bt_names} if is_portfolio else {})}
            all_rows.append({
                "job": name,
                "params": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in full_params.items()},
                "IS_score": r["score"],
                **_row("long", st),
            })
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{n_top}", flush=True)

    if not all_rows:
        raise SystemExit("no valid long-period results")

    df = pd.DataFrame(all_rows).sort_values("long_Sharpe", ascending=False).reset_index(drop=True)
    df.to_csv("bt/longperiod_best5pct.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 80)
    print(f"\n================ LONG-PERIOD ({LONG_START}..{LONG_END}) BEST-OF-TOP-5% ================")
    cols = ["job", "long_return%", "long_CAGR%", "long_Sharpe", "long_Sortino",
            "long_maxDD%", "long_PF", "long_win%", "long_trades", "long_hold"]
    print(df[cols].head(20).to_string(index=False))
    best = df.iloc[0]
    print(f"\nBEST: {best['job']}  Sharpe={best['long_Sharpe']}  return={best['long_return%']}%  "
          f"maxDD={best['long_maxDD%']}%  trades={best['long_trades']}")
    print(f"params: {best['params']}")
    print(f"\nwrote bt/longperiod_best5pct.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
