"""Walk-forward hyperopt of SwingTrendTuned with FULLY independent per-sector
strategy parameters, not just a per-sector stake weight: each of the 19 sectors
gets its own (lookback, reversal_pct, trail_pct, sl, tp) AND its own stake
multiplier - 19 sectors x 6 dims = 114 total dimensions.

This needed a real change to git_stock/backtest.py: run() previously applied
ONE strategy instance's params to every symbol regardless of sector, with
sector_weights only ever scaling stake size. run() now also accepts a
dict[sector_name, Strategy] for `strategy`, resolving each symbol's OWN
per-sector instance for populate_indicators/entry/exit AND for sl/tp/trail/
cooldown - verified against a control test (tech given a deliberately much
tighter tp/sl than every other sector) that this actually changes per-sector
behaviour, and a regression test that passing a single Strategy still
reproduces byte-identical results to before the change.

Same IS-hyperopt -> OOS-validate shape, realistic costs, min-hold anti-scalp
guard as the rest of this session's hyperopts. Writes ONE JSON blob per sector
(params + weight together) to config/sector_params_swingtrend.json, loadable
via git_stock.sector_strategy.load_sector_strategies().

    ./.venv/bin/python scripts/hyperopt_swingtrend_full_sector.py [--trials 6000]
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
STRAT_SPACE = SwingTrendTuned.space   # lookback, reversal_pct, trail_pct, sl, tp


def _row(tag, st):
    return {
        f"{tag}_return%": st["total_return_pct"], f"{tag}_CAGR%": st["cagr_pct"],
        f"{tag}_Sharpe": st["sharpe"], f"{tag}_Sortino": st["sortino"],
        f"{tag}_maxDD%": st["max_drawdown_pct"], f"{tag}_PF": st["profit_factor"],
        f"{tag}_win%": st["win_rate_pct"], f"{tag}_trades": st["trades"],
        f"{tag}_hold": st["avg_bars_held"],
    }


def _build(full: dict, sectors: list[str]) -> tuple[dict, dict]:
    strategies = {s: SwingTrendTuned({k: full[f"{s}__{k}"] for k in STRAT_SPACE}) for s in sectors}
    weights = {s: full[f"{s}__weight"] for s in sectors}
    return strategies, weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=6000)
    ap.add_argument("--cache", default="stock_data")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--out", default="bt/bakeoff_realistic.csv")
    ap.add_argument("--weights-out", default="config/sector_params_swingtrend.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data = DataFeed(args.cache, cfg.timeframe).load_many(cfg.tickers)
    print(f"{len(data)} tickers loaded")
    last = max(df.index[-1] for df in data.values())
    print(f"data through {last.date()}  ·  IS {IS_START}..{IS_END}  ·  OOS {IS_END}..{last.date()}")

    sectors = sorted(cfg.sectors)
    sector_of = {t: sec for sec, ts in cfg.sectors.items() for t in ts}
    per_sector_space = {}
    for s in sectors:
        for k, v in STRAT_SPACE.items():
            per_sector_space[f"{s}__{k}"] = v
        per_sector_space[f"{s}__weight"] = SECTOR_WEIGHT_RANGE
    print(f"per-sector strategy dims: {list(STRAT_SPACE)} + weight  ·  {len(sectors)} sectors  "
          f"·  {len(per_sector_space)} total dims\n")

    bt = BacktestConfig(initial_balance=cfg.initial_balance, stake_amount=170.0, max_open_trades=30,
        fee=0.003, slippage_pct=0.0005, spread_pct=0.0010, fill_next_open=True, gap_stops=True,
        timeframe=cfg.timeframe, sector_of=sector_of)
    score_fn = OBJECTIVES[OBJ]
    rng = random.Random(1)

    t0 = time.time()
    print(f"=== swingtrendfullsector: hyperopt {args.trials} trials on {IS_START}..{IS_END} "
          f"({len(per_sector_space)} dims, objective={OBJ}) ===", flush=True)
    rows = []
    for i in range(args.trials):
        full = _sample(per_sector_space, rng)
        strategies, weights = _build(full, sectors)
        trial_cfg = dataclasses.replace(bt, sector_weights=weights)
        try:
            res = run(data, strategies, trial_cfg, IS_START, IS_END)
        except Exception:
            continue
        st = res.stats
        penalised = st["trades"] < 40
        rows.append({"full": full, **st, "score": (-9e9 if penalised else score_fn(st))})
        if (i + 1) % max(1, args.trials // 20) == 0:
            best = max(r["score"] for r in rows)
            print(f"  {i + 1}/{args.trials}  best {OBJ}={best:.3f}", flush=True)

    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    table = table[table["avg_bars_held"] >= MIN_HOLD]
    if table.empty:
        raise SystemExit("no valid configs above min-hold")

    best_oos, best_val, best_full = None, -1e18, None
    for _, r in table.head(TOP_N).iterrows():
        full = r["full"]
        strategies, weights = _build(full, sectors)
        trial_cfg = dataclasses.replace(bt, sector_weights=weights)
        try:
            oos = run(data, strategies, trial_cfg, OOS_START, OOS_END).stats
        except Exception:
            continue
        bad = oos["trades"] < 20 or oos["avg_bars_held"] < MIN_HOLD
        v = -1e18 if bad else score_fn(oos)
        if v > best_val:
            best_val, best_oos, best_full = v, oos, full
    if best_full is None:
        raise SystemExit("no OOS-valid config")

    strategies, weights = _build(best_full, sectors)
    best_cfg = dataclasses.replace(bt, sector_weights=weights)
    is_stats = run(data, strategies, best_cfg, IS_START, IS_END).stats

    per_sector_out = {s: {**{k: round(best_full[f"{s}__{k}"], 3) for k in STRAT_SPACE},
                          "weight": round(best_full[f"{s}__weight"], 3)} for s in sectors}
    row = {"strategy": "swingtrendfullsector", "params": {"sector_params": per_sector_out},
           **_row("IS", is_stats), **_row("OOS", best_oos)}

    print(f"\nbest OOS: return {best_oos['total_return_pct']}%  Sharpe {best_oos['sharpe']}  "
          f"Sortino {best_oos['sortino']}  maxDD {best_oos['max_drawdown_pct']}%  "
          f"hold {best_oos['avg_bars_held']}b  trades {best_oos['trades']}  ({time.time()-t0:.0f}s)")
    print("per-sector params (sorted by weight):")
    for s in sorted(per_sector_out, key=lambda s: -per_sector_out[s]["weight"]):
        p = per_sector_out[s]
        print(f"  {s:12} weight={p['weight']:.2f}  lookback={p['lookback']:.0f}  "
              f"reversal_pct={p['reversal_pct']:.1f}  trail_pct={p['trail_pct']:.1f}  "
              f"sl={p['sl']:.1f}  tp={p['tp']:.1f}")

    with open(args.weights_out, "w") as f:
        json.dump({"_comment": "hyperopted by scripts/hyperopt_swingtrend_full_sector.py - "
                              "per-sector strategy params + stake weight, load with "
                              "git_stock.sector_strategy.load_sector_strategies()",
                  **per_sector_out}, f, indent=4)
    print(f"\nwrote {args.weights_out}")

    try:
        existing = pd.read_csv(args.out)
        existing = existing[existing["strategy"] != "swingtrendfullsector"]
        merged = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        merged = pd.DataFrame([row])
    merged = merged.sort_values("OOS_Sharpe", ascending=False)
    merged.to_csv(args.out, index=False)

    pd.set_option("display.width", 240)
    cols = [c for c in ["strategy", "OOS_return%", "OOS_CAGR%", "OOS_Sharpe", "OOS_Sortino",
                        "OOS_maxDD%", "OOS_PF", "OOS_win%", "OOS_trades", "OOS_hold",
                        "IS_return%", "IS_Sharpe", "IS_maxDD%"] if c in merged.columns]
    print(f"\n---- {args.out} (swingtrendfullsector merged in, sorted by OOS_Sharpe) ----")
    print(merged[cols].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
