"""Re-validate a wider slice of an already-computed IS hyperopt table OOS, with the
realised/unrealised guard — no re-hyperopting needed, just re-uses bt/is_table_*.csv.

    ./.venv/bin/python scripts/rescan_is_table.py bt/is_table_clecitrailsharpe.csv --top 60
"""
from __future__ import annotations

import argparse
import ast

import pandas as pd

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.optimize import OBJECTIVES
from git_stock.strategies.cleci_trail_sharpe import CleciTrailSharpe

OOS_START, OOS_END = "2022-07-01", "2030-01-01"
OBJ = "sharpe_sortino"
MAX_UNREALISED_FRAC = 0.35
MIN_HOLD = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()

    cfg = Config.load("config/config.json")
    data = DataFeed("stock_data", cfg.timeframe).load_many(cfg.tickers)
    bt = BacktestConfig(initial_balance=cfg.initial_balance, stake_amount=cfg.stake_amount,
        max_open_trades=cfg.max_open_trades, fee=0.003, slippage_pct=0.0005,
        spread_pct=0.0010, fill_next_open=True, gap_stops=True, timeframe=cfg.timeframe)
    score = OBJECTIVES[OBJ]

    table = pd.read_csv(args.table)
    param_names = list(CleciTrailSharpe.space)
    print(f"scanning top {args.top} of {len(table)} IS-ranked configs from {args.table}\n")

    passed, best = [], (None, -1e18, None)
    for _, r in table.head(args.top).iterrows():
        params = {k: r[k] for k in param_names}
        oos = run(data, CleciTrailSharpe(params), bt, OOS_START, OOS_END).stats
        ok = (oos["trades"] >= 20 and oos["avg_bars_held"] >= MIN_HOLD
              and oos["unrealised_frac"] <= MAX_UNREALISED_FRAC)
        if ok:
            passed.append((params, oos))
            v = score(oos)
            if v > best[1]:
                best = (oos, v, params)

    print(f"{len(passed)} of {args.top} candidates pass the guard (trades>=20, hold>={MIN_HOLD}b, "
          f"unrealised_frac<={MAX_UNREALISED_FRAC:.0%})\n")
    if not passed:
        print("none passed — try a larger --top, or the guard may be too strict for this strategy family")
        return

    oos, _, params = best
    print(f"best passing OOS: return {oos['total_return_pct']}%  Sharpe {oos['sharpe']}  "
          f"Sortino {oos['sortino']}  maxDD {oos['max_drawdown_pct']}%  trades {oos['trades']}  "
          f"hold {oos['avg_bars_held']}b  [realised {oos['realised_pct']}% / unrealised "
          f"{oos['unrealised_pct']}% ({oos['open_at_end']} open, {oos['unrealised_frac']:.0%})]")
    print(f"params: {params}")

    print("\n---- all passing candidates, best Sharpe first ----")
    rows = []
    for params, oos in passed:
        rows.append({"params": params, "return%": oos["total_return_pct"], "Sharpe": oos["sharpe"],
                     "Sortino": oos["sortino"], "maxDD%": oos["max_drawdown_pct"],
                     "trades": oos["trades"], "hold": oos["avg_bars_held"],
                     "unrealFrac": oos["unrealised_frac"]})
    df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    pd.set_option("display.width", 200)
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
