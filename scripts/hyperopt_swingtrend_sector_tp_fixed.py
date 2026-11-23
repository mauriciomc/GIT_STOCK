"""Same 23-dim search as hyperopt_swingtrend_sector_tuned.py (4 strategy dims:
lookback/reversal_pct/trail_pct/sl, + 19 sector weights) but with `tp` PINNED to
a fixed value passed on the command line instead of searched. Run once per tp
value to map out how the optimizer's best achievable OOS Sharpe/Sortino/maxDD
responds to tp specifically - letting everything else re-optimize freely at
each tp level, rather than freezing sl/lookback at one tp's best values (which
conflates tp's own effect with whatever sl/lookback happened to compensate for
it during the free 24-dim search).

Writes its own output CSV (per --tp, so parallel runs at different tp values
don't race on the same file) - merge them after the sweep finishes.

    ./.venv/bin/python scripts/hyperopt_swingtrend_sector_tp_fixed.py --tp 300 --trials 3000
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time

import pandas as pd

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.optimize import OBJECTIVES, _sample
from git_stock.strategies.swing_trend_tuned import SwingTrendTuned

IS_START, IS_END = "2018-01-01", "2022-06-30"
OOS_START, OOS_END = "2022-07-01", "2030-01-01"
OBJ = "sharpe_sortino"
TOP_N = 15
MIN_HOLD = 2.0
SECTOR_WEIGHT_RANGE = (0.3, 2.5, 1.0, 0.25)
FREE_SPACE = {k: v for k, v in SwingTrendTuned.space.items() if k != "tp"}


def _row(tag, st):
    return {
        f"{tag}_return%": st["total_return_pct"], f"{tag}_CAGR%": st["cagr_pct"],
        f"{tag}_Sharpe": st["sharpe"], f"{tag}_Sortino": st["sortino"],
        f"{tag}_maxDD%": st["max_drawdown_pct"], f"{tag}_PF": st["profit_factor"],
        f"{tag}_win%": st["win_rate_pct"], f"{tag}_trades": st["trades"],
        f"{tag}_hold": st["avg_bars_held"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=float, required=True)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--cache", default="stock_data")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or f"bt/tp_sweep_{int(args.tp)}.csv"

    cfg = Config.load(args.config)
    data = DataFeed(args.cache, cfg.timeframe).load_many(cfg.tickers)
    sectors = sorted(cfg.sectors)
    sector_of = {t: sec for sec, ts in cfg.sectors.items() for t in ts}
    sector_space = {f"w__{s}": SECTOR_WEIGHT_RANGE for s in sectors}
    combined_space = {**FREE_SPACE, **sector_space}

    bt = BacktestConfig(initial_balance=cfg.initial_balance, stake_amount=170.0, max_open_trades=30,
        fee=0.003, slippage_pct=0.0005, spread_pct=0.0010, fill_next_open=True, gap_stops=True,
        timeframe=cfg.timeframe, sector_of=sector_of)
    score_fn = OBJECTIVES[OBJ]
    rng = random.Random(1)

    t0 = time.time()
    print(f"=== swingtrend tp={args.tp} fixed: hyperopt {args.trials} trials "
          f"(4 strategy + {len(sectors)} sector dims, tp pinned) ===", flush=True)
    rows = []
    for i in range(args.trials):
        full = _sample(combined_space, rng)
        strat_params = {k: full[k] for k in FREE_SPACE}
        strat_params["tp"] = args.tp
        weights = {s: full[f"w__{s}"] for s in sectors}
        trial_cfg = dataclasses.replace(bt, sector_weights=weights)
        try:
            res = run(data, SwingTrendTuned(strat_params), trial_cfg, IS_START, IS_END)
        except Exception:
            continue
        st = res.stats
        penalised = st["trades"] < 40
        rows.append({**strat_params, "weights": weights, **st,
                    "score": (-9e9 if penalised else score_fn(st))})
        if (i + 1) % max(1, args.trials // 10) == 0:
            best = max(r["score"] for r in rows)
            print(f"  tp={args.tp}  {i + 1}/{args.trials}  best {OBJ}={best:.3f}", flush=True)

    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    table = table[table["avg_bars_held"] >= MIN_HOLD]
    if table.empty:
        print(f"tp={args.tp}: no valid configs above min-hold")
        return

    best_oos, best_val, best_strat, best_weights = None, -1e18, None, None
    for _, r in table.head(TOP_N).iterrows():
        strat_params = {k: r[k] for k in FREE_SPACE}
        strat_params["tp"] = args.tp
        trial_cfg = dataclasses.replace(bt, sector_weights=r["weights"])
        oos = run(data, SwingTrendTuned(strat_params), trial_cfg, OOS_START, OOS_END).stats
        bad = oos["trades"] < 20 or oos["avg_bars_held"] < MIN_HOLD
        v = -1e18 if bad else score_fn(oos)
        if v > best_val:
            best_val, best_oos, best_strat, best_weights = v, oos, strat_params, r["weights"]
    if best_strat is None:
        print(f"tp={args.tp}: no OOS-valid config")
        return

    best_cfg = dataclasses.replace(bt, sector_weights=best_weights)
    is_stats = run(data, SwingTrendTuned(best_strat), best_cfg, IS_START, IS_END).stats

    row = {"tp": args.tp,
           "params": {**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_strat.items()},
                     "sector_weights": {k: round(v, 3) for k, v in best_weights.items()}},
           **_row("IS", is_stats), **_row("OOS", best_oos)}

    print(f"\ntp={args.tp}  best OOS: return {best_oos['total_return_pct']}%  Sharpe {best_oos['sharpe']}  "
          f"Sortino {best_oos['sortino']}  maxDD {best_oos['max_drawdown_pct']}%  "
          f"hold {best_oos['avg_bars_held']}b  trades {best_oos['trades']}  ({time.time()-t0:.0f}s)")
    print(f"strategy params: {best_strat}")

    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
