"""A/B test: does portfolio rotation (close weak positions to fund stronger new signals)
help, holding the strategy's own params fixed? Same OOS window, same realistic cost model,
`clecitrailsharpe`'s already-found params, rotation OFF vs ON.

    ./.venv/bin/python scripts/rotation_ablation.py
"""
from __future__ import annotations

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.strategies.cleci_trail_sharpe import CleciTrailSharpe

OOS_START, OOS_END = "2022-07-01", "2030-01-01"
PARAMS = {"window": 160, "entry_mult": 1.12, "exit_mult": 0.77, "trail": 21.5, "sl": 20.0}


def main() -> None:
    cfg = Config.load("config/config.json")
    data = DataFeed("stock_data", cfg.timeframe).load_many(cfg.tickers)
    print(f"{len(data)} tickers loaded\n")

    base = dict(initial_balance=cfg.initial_balance, stake_amount=cfg.stake_amount,
                max_open_trades=cfg.max_open_trades, fee=0.003, slippage_pct=0.0005,
                spread_pct=0.0010, fill_next_open=True, gap_stops=True, timeframe=cfg.timeframe)

    for label, extra in [
        ("cap=40 (never binds) rotation OFF", {}),
        ("cap=40 (never binds) rotation ON", dict(rotate=True, rotate_margin_pct=1.0, min_hold_bars=5)),
        ("cap=15 (binding) rotation OFF", dict(max_open_trades=15)),
        ("cap=15 (binding) rotation ON margin=1.0pp cd=5d", dict(max_open_trades=15, rotate=True, rotate_margin_pct=1.0, min_hold_bars=5)),
        ("cap=15 (binding) rotation ON margin=0.5pp cd=3d", dict(max_open_trades=15, rotate=True, rotate_margin_pct=0.5, min_hold_bars=3)),
        ("cap=10 (binding) rotation OFF", dict(max_open_trades=10)),
        ("cap=10 (binding) rotation ON margin=1.0pp cd=5d", dict(max_open_trades=10, rotate=True, rotate_margin_pct=1.0, min_hold_bars=5)),
    ]:
        bt = BacktestConfig(**{**base, **extra})
        st = run(data, CleciTrailSharpe(PARAMS), bt, OOS_START, OOS_END).stats
        rotated = 0
        res = run(data, CleciTrailSharpe(PARAMS), bt, OOS_START, OOS_END)
        rotated = int((res.trades["exit_reason"] == "rotated_out").sum()) if len(res.trades) else 0
        print(f"{label:42} return {st['total_return_pct']:>7.2f}%  Sharpe {st['sharpe']:>5}  "
              f"Sortino {st['sortino']:>5}  maxDD {st['max_drawdown_pct']:>7.2f}%  "
              f"trades {st['trades']:>4}  hold {st['avg_bars_held']:>6.1f}b  rotated {rotated}")


if __name__ == "__main__":
    main()
