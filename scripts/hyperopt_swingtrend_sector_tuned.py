"""Walk-forward hyperopt of SwingTrendTuned (SwingTrend + a real, searchable
take-profit) jointly with a per-sector stake-size multiplier
(19 sectors + 5 strategy params = 24 dimensions).

Built after finding SwingTrend's own trade log capped every winner at EXACTLY
+98.6% (BacktestConfig's generic 100% default take-profit, never overridden) -
see git_stock/strategies/swing_trend_tuned.py. NVDA's best trade under plain
SwingTrend was cut off at +98.6% despite the stock continuing to run for another
year; this variant lets the optimizer find a real take-profit per the actual
data instead of silently inheriting one it never asked for.

Bespoke trial loop for the same reason as hyperopt_swingtrend_sector.py: 19
individually-sampled scalars need to be ASSEMBLED into the single dict-valued
sector_weights field. Same IS-hyperopt -> OOS-validate shape, realistic costs,
min-hold anti-scalp guard as the rest of this session's hyperopts. Writes to
config/sector_weights_swingtrend_tuned.json (kept separate from the untuned
swingtrendsector weights) plus the usual bt/bakeoff_realistic.csv row.

    ./.venv/bin/python scripts/hyperopt_swingtrend_sector_tuned.py [--trials 3000]
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
SECTOR_WEIGHT_RANGE = (0.3, 2.5, 1.0, 0.25)   # floor raised from 0.0, same reasoning as before


def _row(tag, st):
    return {
        f"{tag}_return%": st["total_return_pct"], f"{tag}_CAGR%": st["cagr_pct"],
        f"{tag}_Sharpe": st["sharpe"], f"{tag}_Sortino": st["sortino"],
        f"{tag}_maxDD%": st["max_drawdown_pct"], f"{tag}_PF": st["profit_factor"],
        f"{tag}_win%": st["win_rate_pct"], f"{tag}_trades": st["trades"],
        f"{tag}_hold": st["avg_bars_held"], f"{tag}_realised%": st["realised_pct"],
        f"{tag}_unrealised%": st["unrealised_pct"], f"{tag}_openAtEnd": st["open_at_end"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--cache", default="stock_data")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--out", default="bt/bakeoff_realistic.csv")
    ap.add_argument("--weights-out", default="config/sector_weights_swingtrend_tuned.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data = DataFeed(args.cache, cfg.timeframe).load_many(cfg.tickers)
    print(f"{len(data)} tickers loaded")
    last = max(df.index[-1] for df in data.values())
    print(f"data through {last.date()}  ·  IS {IS_START}..{IS_END}  ·  OOS {IS_END}..{last.date()}")

    sectors = sorted(cfg.sectors)
    sector_of = {t: sec for sec, ts in cfg.sectors.items() for t in ts}
    sector_space = {f"w__{s}": SECTOR_WEIGHT_RANGE for s in sectors}
    combined_space = {**SwingTrendTuned.space, **sector_space}
    print(f"strategy dims: {list(SwingTrendTuned.space)}")
    print(f"sector dims ({len(sectors)}): {sectors}\n")

    bt = BacktestConfig(initial_balance=cfg.initial_balance, stake_amount=170.0, max_open_trades=30,
        fee=0.003, slippage_pct=0.0005, spread_pct=0.0010, fill_next_open=True, gap_stops=True,
        timeframe=cfg.timeframe, sector_of=sector_of)
    score_fn = OBJECTIVES[OBJ]
    rng = random.Random(1)

    t0 = time.time()
    print(f"=== swingtrendsectortuned: hyperopt {args.trials} trials on {IS_START}..{IS_END} "
          f"(5 strategy + {len(sectors)} sector-weight dims, objective={OBJ}) ===", flush=True)
    rows = []
    for i in range(args.trials):
        full = _sample(combined_space, rng)
        strat_params = {k: full[k] for k in SwingTrendTuned.space}
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
        if (i + 1) % max(1, args.trials // 20) == 0:
            best = max(r["score"] for r in rows)
            print(f"  {i + 1}/{args.trials}  best {OBJ}={best:.3f}", flush=True)

    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    table = table[table["avg_bars_held"] >= MIN_HOLD]
    if table.empty:
        raise SystemExit("no valid configs above min-hold")

    best_oos, best_val, best_strat, best_weights = None, -1e18, None, None
    for _, r in table.head(TOP_N).iterrows():
        strat_params = {k: r[k] for k in SwingTrendTuned.space}
        trial_cfg = dataclasses.replace(bt, sector_weights=r["weights"])
        oos = run(data, SwingTrendTuned(strat_params), trial_cfg, OOS_START, OOS_END).stats
        bad = oos["trades"] < 20 or oos["avg_bars_held"] < MIN_HOLD
        v = -1e18 if bad else score_fn(oos)
        if v > best_val:
            best_val, best_oos, best_strat, best_weights = v, oos, strat_params, r["weights"]
    if best_strat is None:
        raise SystemExit("no OOS-valid config")

    best_cfg = dataclasses.replace(bt, sector_weights=best_weights)
    is_stats = run(data, SwingTrendTuned(best_strat), best_cfg, IS_START, IS_END).stats

    row = {"strategy": "swingtrendsectortuned",
           "params": {**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_strat.items()},
                     "sector_weights": {k: round(v, 3) for k, v in best_weights.items()}},
           **_row("IS", is_stats), **_row("OOS", best_oos)}

    print(f"\nbest OOS: return {best_oos['total_return_pct']}%  Sharpe {best_oos['sharpe']}  "
          f"Sortino {best_oos['sortino']}  maxDD {best_oos['max_drawdown_pct']}%  "
          f"hold {best_oos['avg_bars_held']}b  trades {best_oos['trades']}  ({time.time()-t0:.0f}s)")
    print(f"strategy params: {best_strat}")
    print("sector weights (sorted):")
    for s, w in sorted(best_weights.items(), key=lambda kv: -kv[1]):
        print(f"  {s:12} {w:.2f}")

    with open(args.weights_out, "w") as f:
        json.dump({"_comment": "hyperopted by scripts/hyperopt_swingtrend_sector_tuned.py",
                  **{k: round(v, 3) for k, v in best_weights.items()}}, f, indent=4)
    print(f"\nwrote {args.weights_out}")

    try:
        existing = pd.read_csv(args.out)
        existing = existing[existing["strategy"] != "swingtrendsectortuned"]
        merged = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        merged = pd.DataFrame([row])
    merged = merged.sort_values("OOS_Sharpe", ascending=False)
    merged.to_csv(args.out, index=False)

    pd.set_option("display.width", 240)
    cols = [c for c in ["strategy", "OOS_return%", "OOS_CAGR%", "OOS_Sharpe", "OOS_Sortino",
                        "OOS_maxDD%", "OOS_PF", "OOS_win%", "OOS_trades", "OOS_hold",
                        "IS_return%", "IS_Sharpe", "IS_maxDD%"] if c in merged.columns]
    print(f"\n---- {args.out} (swingtrendsectortuned merged in, sorted by OOS_Sharpe) ----")
    print(merged[cols].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
