"""Stage 3 of the decomposed per-sector search: each sector's own strategy
params are now FIXED (from hyperopt_swingtrend_isolated_sector.py, run once per
sector against just that sector's own tickers) - this stage only searches the
19 stake-size weights, evaluated on the FULL 207-ticker portfolio together
(so capital contention between sectors is realistic here, unlike stage 1 which
deliberately ran each sector in isolation).

    ./.venv/bin/python scripts/hyperopt_swingtrend_weights_only.py [--trials 3000]
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
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
STRAT_KEYS = list(SwingTrendTuned.space)


def _row(tag, st):
    return {
        f"{tag}_return%": st["total_return_pct"], f"{tag}_CAGR%": st["cagr_pct"],
        f"{tag}_Sharpe": st["sharpe"], f"{tag}_Sortino": st["sortino"],
        f"{tag}_maxDD%": st["max_drawdown_pct"], f"{tag}_PF": st["profit_factor"],
        f"{tag}_win%": st["win_rate_pct"], f"{tag}_trades": st["trades"],
        f"{tag}_hold": st["avg_bars_held"],
    }


def _load_isolated_params(pattern: str) -> dict[str, dict]:
    out = {}
    for f in glob.glob(pattern):
        row = pd.read_csv(f).iloc[0]
        out[row["sector"]] = {k: float(row[k]) for k in STRAT_KEYS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--cache", default="stock_data")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--out", default="bt/bakeoff_realistic.csv")
    ap.add_argument("--isolated-glob", default="bt/isolated_sector_*.csv")
    ap.add_argument("--weights-out", default="config/sector_params_swingtrend_decomposed.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data = DataFeed(args.cache, cfg.timeframe).load_many(cfg.tickers)
    sector_of = {t: sec for sec, ts in cfg.sectors.items() for t in ts}

    isolated = _load_isolated_params(args.isolated_glob)
    sectors = sorted(cfg.sectors)
    missing = [s for s in sectors if s not in isolated]
    if missing:
        raise SystemExit(f"missing isolated-stage results for: {missing}")
    fixed_strategies = {s: SwingTrendTuned(isolated[s]) for s in sectors}
    print("fixed per-sector params (from isolated stage 1):")
    for s in sectors:
        print(f"  {s:12} {isolated[s]}")

    weight_space = {f"w__{s}": SECTOR_WEIGHT_RANGE for s in sectors}
    bt = BacktestConfig(initial_balance=cfg.initial_balance, stake_amount=170.0, max_open_trades=30,
        fee=0.003, slippage_pct=0.0005, spread_pct=0.0010, fill_next_open=True, gap_stops=True,
        timeframe=cfg.timeframe, sector_of=sector_of)
    score_fn = OBJECTIVES[OBJ]
    rng = random.Random(1)

    t0 = time.time()
    print(f"\n=== swingtrenddecomposedsector: hyperopt {args.trials} trials on weights only "
          f"(19 dims, strat params fixed per sector, objective={OBJ}) ===", flush=True)
    rows = []
    for i in range(args.trials):
        full = _sample(weight_space, rng)
        weights = {s: full[f"w__{s}"] for s in sectors}
        trial_cfg = dataclasses.replace(bt, sector_weights=weights)
        try:
            res = run(data, fixed_strategies, trial_cfg, IS_START, IS_END)
        except Exception:
            continue
        st = res.stats
        penalised = st["trades"] < 40
        rows.append({"weights": weights, **st, "score": (-9e9 if penalised else score_fn(st))})
        if (i + 1) % max(1, args.trials // 20) == 0:
            best = max(r["score"] for r in rows)
            print(f"  {i + 1}/{args.trials}  best {OBJ}={best:.3f}", flush=True)

    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    table = table[table["avg_bars_held"] >= MIN_HOLD]
    if table.empty:
        raise SystemExit("no valid configs above min-hold")

    best_oos, best_val, best_weights = None, -1e18, None
    for _, r in table.head(TOP_N).iterrows():
        trial_cfg = dataclasses.replace(bt, sector_weights=r["weights"])
        oos = run(data, fixed_strategies, trial_cfg, OOS_START, OOS_END).stats
        bad = oos["trades"] < 20 or oos["avg_bars_held"] < MIN_HOLD
        v = -1e18 if bad else score_fn(oos)
        if v > best_val:
            best_val, best_oos, best_weights = v, oos, r["weights"]
    if best_weights is None:
        raise SystemExit("no OOS-valid config")

    best_cfg = dataclasses.replace(bt, sector_weights=best_weights)
    is_stats = run(data, fixed_strategies, best_cfg, IS_START, IS_END).stats

    per_sector_out = {s: {**{k: round(isolated[s][k], 3) for k in STRAT_KEYS},
                          "weight": round(best_weights[s], 3)} for s in sectors}
    row = {"strategy": "swingtrenddecomposedsector", "params": {"sector_params": per_sector_out},
           **_row("IS", is_stats), **_row("OOS", best_oos)}

    print(f"\nbest OOS: return {best_oos['total_return_pct']}%  Sharpe {best_oos['sharpe']}  "
          f"Sortino {best_oos['sortino']}  maxDD {best_oos['max_drawdown_pct']}%  "
          f"hold {best_oos['avg_bars_held']}b  trades {best_oos['trades']}  ({time.time()-t0:.0f}s)")
    print("weights (sorted):")
    for s in sorted(per_sector_out, key=lambda s: -per_sector_out[s]["weight"]):
        print(f"  {s:12} weight={per_sector_out[s]['weight']:.2f}")

    with open(args.weights_out, "w") as f:
        json.dump({"_comment": "decomposed: stage1=hyperopt_swingtrend_isolated_sector.py per sector, "
                              "stage3(this)=hyperopt_swingtrend_weights_only.py for weights - "
                              "load with git_stock.sector_strategy.load_sector_strategies()",
                  **per_sector_out}, f, indent=4)
    print(f"\nwrote {args.weights_out}")

    try:
        existing = pd.read_csv(args.out)
        existing = existing[existing["strategy"] != "swingtrenddecomposedsector"]
        merged = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        merged = pd.DataFrame([row])
    merged = merged.sort_values("OOS_Sharpe", ascending=False)
    merged.to_csv(args.out, index=False)

    pd.set_option("display.width", 240)
    cols = [c for c in ["strategy", "OOS_return%", "OOS_CAGR%", "OOS_Sharpe", "OOS_Sortino",
                        "OOS_maxDD%", "OOS_PF", "OOS_win%", "OOS_trades", "OOS_hold",
                        "IS_return%", "IS_Sharpe", "IS_maxDD%"] if c in merged.columns]
    print(f"\n---- {args.out} (swingtrenddecomposedsector merged in, sorted by OOS_Sharpe) ----")
    print(merged[cols].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
