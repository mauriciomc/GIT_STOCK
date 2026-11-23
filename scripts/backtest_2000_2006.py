"""Test today's already-settled strategy configs on a completely different, non-overlapping
regime: 2000-01-01..2006-12-31 (dot-com bust + recovery) instead of 2018-2026. Not
re-hyperopted — this checks whether the SAME parameters found on 2018-2022/2022-2026 hold
up on a market regime they've never seen, using the freshly fetched stock_data_full/ cache
(128 of 220 tickers have data back to 2000).

    ./.venv/bin/python scripts/backtest_2000_2006.py
"""
from __future__ import annotations

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.strategies import get as get_strategy
from git_stock.strategies.cleci_trail_sharpe import CleciTrailSharpe

START, END = "2000-01-01", "2006-12-31"

CASES = [
    ("cleci (Run3 realistic winner)", get_strategy("cleci"),
     {"window": 210, "entry_mult": 1.09, "exit_mult": 0.82, "use_rolling": True,
      "high_band": "high", "exit_mode": "trail", "tp": 20.0, "sl": 23.0, "trail": 17.5},
     dict(stake_amount=40, max_open_trades=40)),
    ("clecitrail (Run3 realistic winner)", get_strategy("clecitrail"),
     {"window": 210, "entry_mult": 1.22, "exit_mult": 0.79, "trail": 23.5, "sl": 17.0},
     dict(stake_amount=40, max_open_trades=40)),
    ("clecitrailsharpe (OLD ungated)", CleciTrailSharpe,
     {"window": 160, "entry_mult": 1.12, "exit_mult": 0.77, "trail": 21.5, "sl": 20.0},
     dict(stake_amount=40, max_open_trades=40)),
    ("clecitrailsharpe (NEW guarded champion)", CleciTrailSharpe,
     {"window": 350, "entry_mult": 1.17, "exit_mult": 0.7, "trail": 15.0, "sl": 29.0},
     dict(stake_amount=40, max_open_trades=40)),
    ("clecitrailportfolio (guarded champion)", CleciTrailSharpe,
     {"window": 190, "entry_mult": 1.17, "exit_mult": 0.79, "trail": 16.5, "sl": 30.0},
     dict(stake_amount=125.0, max_open_trades=31, rotate=True, rotate_margin_pct=6.0, min_hold_bars=11)),
]


def main() -> None:
    cfg = Config.load("config/config.json")
    data = DataFeed("stock_data_full", cfg.timeframe).load_many(cfg.tickers)
    usable = sum(1 for df in data.values() if df.index[0] <= __import__("pandas").Timestamp("2000-06-01"))
    print(f"{len(data)} tickers loaded, {usable} with data covering the 2000-2006 window")
    print(f"backtest window: {START}..{END} (dot-com bust + recovery)\n")

    base = dict(initial_balance=cfg.initial_balance, fee=0.003, slippage_pct=0.0005,
                spread_pct=0.0010, fill_next_open=True, gap_stops=True, timeframe=cfg.timeframe)

    rows = []
    for label, cls, params, btextra in CASES:
        bt = BacktestConfig(**{**base, **btextra})
        st = run(data, cls(params), bt, START, END).stats
        rows.append((label, st))
        print(f"{label:42} return {st['total_return_pct']:>8.2f}%  CAGR {st['cagr_pct']:>6.2f}%  "
              f"Sharpe {str(st['sharpe']):>6}  Sortino {str(st['sortino']):>6}  "
              f"maxDD {st['max_drawdown_pct']:>7.2f}%  trades {st['trades']:>4}  "
              f"hold {st['avg_bars_held']:>6.1f}b  [unreal {st['unrealised_frac']:.0%}, "
              f"{st['open_at_end']} open]")


if __name__ == "__main__":
    main()
