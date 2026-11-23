"""Compare CleciRange (the fixed, range-percentage version) against the current
live CleciTrail config, same data, same walk-forward split, same cost model -
a direct answer to whether fixing the threshold-inversion issue actually helps.

    ./.venv/bin/python scripts/compare_clecirange.py
"""
from __future__ import annotations

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.strategies.cleci_range import CleciRange
from git_stock.strategies.cleci_trail_live import LIVE_PARAMS, CleciTrailLive

IS_START, IS_END = "2018-01-01", "2022-06-30"
OOS_START, OOS_END = "2022-07-01", "2030-01-01"


def _print(tag: str, params: dict, is_stats: dict, oos_stats: dict) -> None:
    print(f"\n=== {tag}  params={params} ===")
    print(f"  IS  ({IS_START}..{IS_END}):  return {is_stats['total_return_pct']}%  "
          f"CAGR {is_stats['cagr_pct']}%  Sharpe {is_stats['sharpe']}  "
          f"maxDD {is_stats['max_drawdown_pct']}%  PF {is_stats['profit_factor']}  "
          f"win {is_stats['win_rate_pct']}%  trades {is_stats['trades']}  "
          f"hold {is_stats['avg_bars_held']}b")
    print(f"  OOS ({OOS_START}..present): return {oos_stats['total_return_pct']}%  "
          f"CAGR {oos_stats['cagr_pct']}%  Sharpe {oos_stats['sharpe']}  "
          f"maxDD {oos_stats['max_drawdown_pct']}%  PF {oos_stats['profit_factor']}  "
          f"win {oos_stats['win_rate_pct']}%  trades {oos_stats['trades']}  "
          f"hold {oos_stats['avg_bars_held']}b")


def main() -> None:
    cfg = Config.load("config/config.json")
    data = DataFeed("stock_data", cfg.timeframe).load_many(cfg.tickers)
    print(f"{len(data)} tickers loaded")

    realistic = BacktestConfig(
        initial_balance=cfg.initial_balance, stake_amount=cfg.stake_amount,
        max_open_trades=cfg.max_open_trades, fee=0.003, slippage_pct=0.0005,
        spread_pct=0.0010, fill_next_open=True, gap_stops=True, timeframe=cfg.timeframe)

    # current live config, unchanged, same cost model
    live = CleciTrailLive()
    live_is = run(data, live, realistic, IS_START, IS_END).stats
    live_oos = run(data, live, realistic, OOS_START, OOS_END).stats
    _print("clecitrail_live (CURRENT, unchanged)", LIVE_PARAMS, live_is, live_oos)

    # same window/trail/sl, but the range-percentage thresholds instead of
    # independent multipliers - buy_pct/sell_pct chosen to bracket roughly the
    # same "buy near the low, sell near the high" feel as entry_mult=1.16/exit_mult=0.77
    range_params = {
        "window": LIVE_PARAMS["window"], "buy_pct": 16.0, "sell_pct": 77.0,
        "trail": LIVE_PARAMS["trail"], "sl": LIVE_PARAMS["sl"],
        "exit_mode": "band+trail",
    }
    rng_strat = CleciRange(range_params)
    rng_is = run(data, rng_strat, realistic, IS_START, IS_END).stats
    rng_oos = run(data, rng_strat, realistic, OOS_START, OOS_END).stats
    _print("clecirange (NEW, fixed inversion)", range_params, rng_is, rng_oos)


if __name__ == "__main__":
    main()
