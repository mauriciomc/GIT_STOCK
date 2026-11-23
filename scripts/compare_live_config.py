"""Score the fixed CleciTrailLive config (== config.json's current clecitrail
params) on the same walk-forward split as scripts/bakeoff.py, under both cost
models, and merge it into the existing bake-off CSVs for a direct comparison.

No hyperopt here — one fixed config, backtested IS + OOS, exactly like every
other row in the bake-off tables.

    ./.venv/bin/python scripts/compare_live_config.py
"""
from __future__ import annotations

import pandas as pd

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.strategies.cleci_trail_live import LIVE_PARAMS, CleciTrailLive

IS_START, IS_END = "2018-01-01", "2022-06-30"
OOS_START, OOS_END = "2022-07-01", "2030-01-01"


def _row(tag, st):
    return {
        f"{tag}_return%": st["total_return_pct"], f"{tag}_CAGR%": st["cagr_pct"],
        f"{tag}_Sharpe": st["sharpe"], f"{tag}_maxDD%": st["max_drawdown_pct"],
        f"{tag}_PF": st["profit_factor"], f"{tag}_win%": st["win_rate_pct"],
        f"{tag}_trades": st["trades"], f"{tag}_hold": st["avg_bars_held"],
    }


def score(mode: str, cfg: Config, data: dict, bt: BacktestConfig, csv_path: str) -> dict:
    strat = CleciTrailLive()
    is_stats = run(data, strat, bt, IS_START, IS_END).stats
    oos_stats = run(data, strat, bt, OOS_START, OOS_END).stats
    row = {"strategy": "clecitrail_live", "params": LIVE_PARAMS,
           **_row("IS", is_stats), **_row("OOS", oos_stats)}
    print(f"\n=== clecitrail_live ({mode}) — fixed config.json params {LIVE_PARAMS} ===")
    print(f"  IS  ({IS_START}..{IS_END}):  return {is_stats['total_return_pct']}%  "
          f"Sharpe {is_stats['sharpe']}  maxDD {is_stats['max_drawdown_pct']}%  "
          f"trades {is_stats['trades']}  hold {is_stats['avg_bars_held']}b")
    print(f"  OOS ({OOS_START}..present): return {oos_stats['total_return_pct']}%  "
          f"Sharpe {oos_stats['sharpe']}  maxDD {oos_stats['max_drawdown_pct']}%  "
          f"trades {oos_stats['trades']}  hold {oos_stats['avg_bars_held']}b")

    try:
        existing = pd.read_csv(csv_path)
        existing = existing[existing["strategy"] != "clecitrail_live"]
        merged = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        merged = pd.DataFrame([row])
    merged = merged.sort_values("OOS_return%", ascending=False)
    merged.to_csv(csv_path, index=False)

    pd.set_option("display.width", 240)
    print(f"\n---- {csv_path} (with clecitrail_live merged in) ----")
    print(merged[["strategy", "OOS_return%", "OOS_CAGR%", "OOS_Sharpe", "OOS_maxDD%",
                  "OOS_PF", "OOS_win%", "OOS_trades", "OOS_hold",
                  "IS_return%", "IS_Sharpe", "IS_maxDD%"]].to_string(index=False))
    return row


def main() -> None:
    cfg = Config.load("config/config.json")
    data = DataFeed("stock_data", cfg.timeframe).load_many(cfg.tickers)
    print(f"{len(data)} tickers loaded")

    realistic = BacktestConfig(
        initial_balance=cfg.initial_balance, stake_amount=cfg.stake_amount,
        max_open_trades=cfg.max_open_trades, fee=0.003, slippage_pct=0.0005,
        spread_pct=0.0010, fill_next_open=True, gap_stops=True, timeframe=cfg.timeframe)
    score("realistic", cfg, data, realistic, "bt/bakeoff_realistic.csv")

    frictionless = BacktestConfig(
        initial_balance=cfg.initial_balance, stake_amount=cfg.stake_amount,
        max_open_trades=cfg.max_open_trades, fee=0.001, slippage_pct=0.0,
        spread_pct=0.0, fill_next_open=False, gap_stops=False, timeframe=cfg.timeframe)
    score("frictionless", cfg, data, frictionless, "bt/bakeoff_frictionless.csv")


if __name__ == "__main__":
    main()
