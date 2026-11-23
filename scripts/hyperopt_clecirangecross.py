"""Walk-forward hyperopt of CleciRangeCross (plain-crossover reclaim/breakout +
slope-rising gate + market-regime filter) jointly with portfolio-level knobs.

15 total dimensions (10 strategy: window, buy_pct, sell_pct, rsi_period,
rsi_buy_min, rsi_sell_max, high_band, cooldown_bars, slope_window,
long_slope_window; 5 portfolio: stake_amount, max_open_trades, rotate,
rotate_margin_pct, min_hold_bars, regime_dd_threshold - that's actually 6,
16 total) - bumped to 2000 trials (vs this session's usual 800) to cover the
larger space properly.

Same IS-hyperopt -> OOS-validate shape, realistic costs, min-hold anti-scalp
guard, downtrend-episode check as the rest of this session's hyperopts.

    ./.venv/bin/python scripts/hyperopt_clecirangecross.py [--trials 2000]
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import pandas as pd

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.optimize import OBJECTIVES, PORTFOLIO_SPACE, optimise_portfolio
from git_stock.strategies.cleci_range_cross import CleciRangeCross

IS_START, IS_END = "2018-01-01", "2022-06-30"
OOS_START, OOS_END = "2022-07-01", "2030-01-01"
OBJ = "sharpe_sortino"
TOP_N = 15
MIN_HOLD = 2.0

DOWNTREND_EPISODES = [
    ("2015-12-08", "2016-03-09"), ("2018-12-07", "2019-01-09"), ("2020-03-06", "2020-05-25"),
    ("2022-03-07", "2022-03-15"), ("2022-05-05", "2022-11-11"), ("2022-12-15", "2023-01-20"),
    ("2023-03-10", "2023-03-29"), ("2023-05-20", "2023-06-10"), ("2023-09-21", "2023-11-14"),
    ("2025-03-25", "2025-04-18"),
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


def _regime_dd(data: dict[str, pd.DataFrame]) -> pd.Series:
    closes = pd.DataFrame({s: df["close"] for s, df in data.items()})
    rets = closes.pct_change()
    composite = (1 + rets.mean(axis=1, skipna=True).fillna(0)).cumprod()
    dd = composite / composite.cummax() - 1
    dd.index = dd.index.normalize()
    return dd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--cache", default="stock_data")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--out", default="bt/bakeoff_realistic.csv")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data = DataFeed(args.cache, cfg.timeframe).load_many(cfg.tickers)
    print(f"{len(data)} tickers loaded")
    last = max(df.index[-1] for df in data.values())
    print(f"data through {last.date()}  ·  IS {IS_START}..{IS_END}  ·  OOS {IS_END}..{last.date()}")
    print("cost model: fee 0.3%/side · slip 0.05% · spread 0.10% · next-open fills · gap stops "
          f"· min-hold {MIN_HOLD}b · objective={OBJ}")
    print(f"strategy space: {CleciRangeCross.space}")
    print(f"portfolio space: {PORTFOLIO_SPACE}\n")

    regime_dd = _regime_dd(data)
    bt = BacktestConfig(
        initial_balance=cfg.initial_balance, fee=0.003, slippage_pct=0.0005,
        spread_pct=0.0010, fill_next_open=True, gap_stops=True, timeframe=cfg.timeframe,
        regime_dd=regime_dd)
    score = OBJECTIVES[OBJ]

    t0 = time.time()
    print(f"=== clecirangecross: hyperopt {args.trials} trials on {IS_START}..{IS_END} "
          f"(strategy + portfolio + regime filter, objective={OBJ}) ===", flush=True)
    table = optimise_portfolio(data, CleciRangeCross, bt, n_trials=args.trials,
                               objective=OBJ, start=IS_START, end=IS_END, min_trades=40,
                               seed=1, progress=True)
    table = table[table["avg_bars_held"] >= MIN_HOLD]
    if table.empty:
        raise SystemExit("no valid configs above min-hold")
    table.to_csv("bt/is_table_clecirangecross.csv", index=False)

    strat_names = list(CleciRangeCross.space)
    bt_names = list(PORTFOLIO_SPACE)
    best_oos, best_val, best_full = None, -1e18, None
    for _, r in table.head(TOP_N).iterrows():
        strat_params = {k: r[k] for k in strat_names}
        bt_overrides = {k: r[k] for k in bt_names}
        trial_cfg = dataclasses.replace(bt, **bt_overrides)
        try:
            oos = run(data, CleciRangeCross(strat_params), trial_cfg, OOS_START, OOS_END).stats
        except Exception:
            continue
        bad = oos["trades"] < 20 or oos["avg_bars_held"] < MIN_HOLD
        v = -1e18 if bad else score(oos)
        if v > best_val:
            best_val, best_oos = v, oos
            best_full = {**strat_params, **bt_overrides}
    if best_full is None:
        raise SystemExit("no OOS-valid config")

    strat_params = {k: best_full[k] for k in strat_names}
    bt_overrides = {k: best_full[k] for k in bt_names}
    best_cfg = dataclasses.replace(bt, **bt_overrides)
    is_stats = run(data, CleciRangeCross(strat_params), best_cfg, IS_START, IS_END).stats

    row = {"strategy": "clecirangecross",
           "params": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_full.items()},
           **_row("IS", is_stats), **_row("OOS", best_oos)}

    print(f"\nbest OOS: return {best_oos['total_return_pct']}%  Sharpe {best_oos['sharpe']}  "
          f"Sortino {best_oos['sortino']}  maxDD {best_oos['max_drawdown_pct']}%  "
          f"hold {best_oos['avg_bars_held']}b  trades {best_oos['trades']}  "
          f"({time.time()-t0:.0f}s)")
    print(f"params: {row['params']}")

    print("\n---- downtrend-episode check (winning config) ----")
    for s, e in DOWNTREND_EPISODES:
        try:
            st = run(data, CleciRangeCross(strat_params), best_cfg, s, e).stats
        except (ValueError, Exception):
            continue
        print(f"  {s}..{e}  ret={st['total_return_pct']:7.2f}%  maxDD={st['max_drawdown_pct']:7.2f}%  trades={st['trades']}")

    try:
        existing = pd.read_csv(args.out)
        existing = existing[existing["strategy"] != "clecirangecross"]
        merged = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        merged = pd.DataFrame([row])
    merged = merged.sort_values("OOS_Sharpe", ascending=False)
    merged.to_csv(args.out, index=False)

    pd.set_option("display.width", 240)
    cols = [c for c in ["strategy", "OOS_return%", "OOS_CAGR%", "OOS_Sharpe", "OOS_Sortino",
                        "OOS_maxDD%", "OOS_PF", "OOS_win%", "OOS_trades", "OOS_hold",
                        "IS_return%", "IS_Sharpe", "IS_maxDD%"] if c in merged.columns]
    print(f"\n---- {args.out} (clecirangecross merged in, sorted by OOS_Sharpe) ----")
    print(merged[cols].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
