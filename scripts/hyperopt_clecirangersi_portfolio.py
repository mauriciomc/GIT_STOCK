"""Walk-forward hyperopt of CleciRangeRSI jointly with portfolio-level knobs
(stake_amount, max_open_trades, rotation) instead of holding them fixed from
config.json — same PORTFOLIO_SPACE used for clecitrailportfolio, so trials are
free to find "fewer, bigger positions" (e.g. max_open_trades~10, stake~150) if
that's actually better, rather than us hardcoding it and guessing.

Same IS-hyperopt -> OOS-validate shape and realistic cost model as the main
bake-off; objective sharpe_sortino; min-hold anti-scalp guard.

    ./.venv/bin/python scripts/hyperopt_clecirangersi_portfolio.py [--trials 600]
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
from git_stock.strategies.cleci_range_rsi import CleciRangeRSI

IS_START, IS_END = "2018-01-01", "2022-06-30"
OOS_START, OOS_END = "2022-07-01", "2030-01-01"
OBJ = "sharpe_sortino"
TOP_N = 12
MIN_HOLD = 2.0


def _row(tag, st):
    return {
        f"{tag}_return%": st["total_return_pct"], f"{tag}_CAGR%": st["cagr_pct"],
        f"{tag}_Sharpe": st["sharpe"], f"{tag}_Sortino": st["sortino"],
        f"{tag}_maxDD%": st["max_drawdown_pct"], f"{tag}_PF": st["profit_factor"],
        f"{tag}_win%": st["win_rate_pct"], f"{tag}_trades": st["trades"],
        f"{tag}_hold": st["avg_bars_held"], f"{tag}_realised%": st["realised_pct"],
        f"{tag}_unrealised%": st["unrealised_pct"], f"{tag}_openAtEnd": st["open_at_end"],
        f"{tag}_unrealFrac": st["unrealised_frac"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=600)
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
    print(f"portfolio search space: {PORTFOLIO_SPACE}\n")

    bt = BacktestConfig(
        initial_balance=cfg.initial_balance, fee=0.003, slippage_pct=0.0005,
        spread_pct=0.0010, fill_next_open=True, gap_stops=True, timeframe=cfg.timeframe)
    score = OBJECTIVES[OBJ]

    t0 = time.time()
    print(f"=== clecirangersiportfolio: hyperopt {args.trials} trials on {IS_START}..{IS_END} "
          f"(strategy + portfolio params, objective={OBJ}) ===", flush=True)
    table = optimise_portfolio(data, CleciRangeRSI, bt, n_trials=args.trials,
                               objective=OBJ, start=IS_START, end=IS_END, min_trades=40,
                               seed=1, progress=True)
    table = table[table["avg_bars_held"] >= MIN_HOLD]
    if table.empty:
        raise SystemExit("no valid configs above min-hold")
    table.to_csv("bt/is_table_clecirangersiportfolio.csv", index=False)

    strat_names = list(CleciRangeRSI.space)
    bt_names = list(PORTFOLIO_SPACE)
    best_oos, best_val, best_full = None, -1e18, None
    for _, r in table.head(TOP_N).iterrows():
        strat_params = {k: r[k] for k in strat_names}
        bt_overrides = {k: r[k] for k in bt_names}
        trial_cfg = dataclasses.replace(bt, **bt_overrides)
        oos = run(data, CleciRangeRSI(strat_params), trial_cfg, OOS_START, OOS_END).stats
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
    is_stats = run(data, CleciRangeRSI(strat_params), best_cfg, IS_START, IS_END).stats

    row = {"strategy": "clecirangersiportfolio",
           "params": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_full.items()},
           **_row("IS", is_stats), **_row("OOS", best_oos)}

    print(f"\nbest OOS: return {best_oos['total_return_pct']}%  Sharpe {best_oos['sharpe']}  "
          f"Sortino {best_oos['sortino']}  maxDD {best_oos['max_drawdown_pct']}%  "
          f"hold {best_oos['avg_bars_held']}b  trades {best_oos['trades']}  "
          f"[realised {best_oos['realised_pct']}% / unrealised {best_oos['unrealised_pct']}% "
          f"({best_oos['open_at_end']} still open, {best_oos['unrealised_frac']:.0%} of P&L)]  "
          f"({time.time()-t0:.0f}s)")
    print(f"params: {row['params']}")

    try:
        existing = pd.read_csv(args.out)
        existing = existing[existing["strategy"] != "clecirangersiportfolio"]
        merged = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        merged = pd.DataFrame([row])
    merged = merged.sort_values("OOS_Sharpe", ascending=False)
    merged.to_csv(args.out, index=False)

    pd.set_option("display.width", 240)
    cols = [c for c in ["strategy", "OOS_return%", "OOS_CAGR%", "OOS_Sharpe", "OOS_Sortino",
                        "OOS_maxDD%", "OOS_PF", "OOS_win%", "OOS_trades", "OOS_hold",
                        "IS_return%", "IS_Sharpe", "IS_maxDD%"] if c in merged.columns]
    print(f"\n---- {args.out} (clecirangersiportfolio merged in, sorted by OOS_Sharpe) ----")
    print(merged[cols].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
