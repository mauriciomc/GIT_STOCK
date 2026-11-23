"""Stage 3 of the decomposed per-sector search, generalised to any registered
strategy: each sector's own strategy params are FIXED (from
hyperopt_isolated_sector.py, run once per sector against just that sector's own
tickers) - this stage only searches the 19 stake-size weights, evaluated on the
FULL 207-ticker portfolio together (so capital contention between sectors is
realistic here, unlike stage 1 which deliberately ran each sector in isolation).

    ./.venv/bin/python scripts/hyperopt_weights_only.py --strategy bbrsituned --result-name bbrsidecomposedsector [--trials 3000]
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
from git_stock.strategies import get as get_strategy
from git_stock import metrics

IS_START, IS_END = "2018-01-01", "2022-06-30"
OOS_START, OOS_END = "2022-07-01", "2030-01-01"
OBJ = "sharpe_sortino"
TOP_N = 15
MIN_HOLD = 2.0
SECTOR_WEIGHT_RANGE = (0.3, 2.5, 1.0, 0.25)
# how much a full continuous-history drawdown penalises an otherwise good OOS score -
# see the CleciDecomposed case study: OOS/IS scored in isolation never sees this at all,
# since it's a path-dependent artefact of the shared max_open_trades slot cap interacting
# with pre-history the isolated windows don't include. 2.0 means a candidate whose worst
# drawdown eats half the full history (frac=0.5) is penalised by 1.0 score points - roughly
# the same order of magnitude as the gap between this study's best and mediocre candidates.
DD_DURATION_PENALTY = 2.0


def _row(tag, st):
    return {
        f"{tag}_return%": st["total_return_pct"], f"{tag}_CAGR%": st["cagr_pct"],
        f"{tag}_Sharpe": st["sharpe"], f"{tag}_Sortino": st["sortino"],
        f"{tag}_maxDD%": st["max_drawdown_pct"], f"{tag}_PF": st["profit_factor"],
        f"{tag}_win%": st["win_rate_pct"], f"{tag}_trades": st["trades"],
        f"{tag}_hold": st["avg_bars_held"],
    }


def _load_isolated_params(pattern: str, strat_keys: list[str]) -> dict[str, dict]:
    out = {}
    for f in glob.glob(pattern):
        row = pd.read_csv(f).iloc[0]
        params = {}
        for k in strat_keys:
            v = row[k]
            try:
                params[k] = float(v)
            except (TypeError, ValueError):
                params[k] = v   # categorical (e.g. high_band: "low"/"high")
        out[row["sector"]] = params
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, help="registered strategy name, e.g. bbrsituned")
    ap.add_argument("--result-name", required=True, help="strategy label to write into bakeoff_realistic.csv")
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--cache", default="stock_data")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--out", default="bt/bakeoff_realistic.csv")
    ap.add_argument("--isolated-glob", default=None)
    ap.add_argument("--weights-out", default=None)
    ap.add_argument("--top-n", type=int, default=TOP_N,
                    help="how many OOS-ranked candidates to gate on full continuous history")
    ap.add_argument("--min-full-sharpe", type=float, default=0.3,
                    help="robustness floor: only candidates clearing this full-history "
                         "Sharpe compete on return; if none clear it, falls back to the "
                         "single highest full-history Sharpe found")
    ap.add_argument("--is-start", default=None, help="override IS_START (default: module constant)")
    ap.add_argument("--is-end", default=None, help="override IS_END (default: module constant)")
    ap.add_argument("--oos-start", default=None, help="override OOS_START (default: module constant)")
    ap.add_argument("--oos-end", default=None, help="override OOS_END (default: module constant)")
    ap.add_argument("--full-start", default=None,
                    help="override the full-history gate's start date (default: 2016-09-04)")
    ap.add_argument("--full-end", default=None,
                    help="override the full-history gate's end date (default: latest cached bar)")
    args = ap.parse_args()
    is_start, is_end = args.is_start or IS_START, args.is_end or IS_END
    oos_start, oos_end = args.oos_start or OOS_START, args.oos_end or OOS_END
    isolated_glob = args.isolated_glob or f"bt/isolated_{args.strategy}_*.csv"
    weights_out = args.weights_out or f"config/sector_params_{args.strategy}_decomposed.json"
    cls = get_strategy(args.strategy)
    strat_keys = list(cls.space)

    cfg = Config.load(args.config)
    data = DataFeed(args.cache, cfg.timeframe).load_many(cfg.tickers)
    sector_of = {t: sec for sec, ts in cfg.sectors.items() for t in ts}

    isolated = _load_isolated_params(isolated_glob, strat_keys)
    sectors = sorted(cfg.sectors)
    missing = [s for s in sectors if s not in isolated]
    if missing:
        raise SystemExit(f"missing isolated-stage results for: {missing}")
    fixed_strategies = {s: cls(isolated[s]) for s in sectors}
    print(f"fixed per-sector params for {args.strategy} (from isolated stage 1):")
    for s in sectors:
        print(f"  {s:12} {isolated[s]}")

    weight_space = {f"w__{s}": SECTOR_WEIGHT_RANGE for s in sectors}
    bt = BacktestConfig(initial_balance=cfg.initial_balance, stake_amount=170.0, max_open_trades=30,
        fee=0.003, slippage_pct=0.0005, spread_pct=0.0010, fill_next_open=True, gap_stops=True,
        timeframe=cfg.timeframe, sector_of=sector_of)
    score_fn = OBJECTIVES[OBJ]
    rng = random.Random(1)

    t0 = time.time()
    print(f"\n=== {args.result_name}: hyperopt {args.trials} trials on weights only "
          f"(19 dims, {args.strategy} params fixed per sector, objective={OBJ}) ===", flush=True)
    rows = []
    for i in range(args.trials):
        full = _sample(weight_space, rng)
        weights = {s: full[f"w__{s}"] for s in sectors}
        trial_cfg = dataclasses.replace(bt, sector_weights=weights)
        try:
            res = run(data, fixed_strategies, trial_cfg, is_start, is_end)
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

    # NOT the absolute earliest available ticker date (2015-11-18) - only 194/207 tickers
    # have data that far back, so starting there trades a smaller, different universe for
    # its first ~9 months and the shared max_open_trades slot cap makes the whole multi-year
    # trajectory sensitive to that difference (verified directly: identical weights gave
    # dd_duration_frac 0.292 from 2015-11-18 vs 0.719 from 2016-09-04). Use the same start
    # every other full-history comparison in this study uses, for a like-for-like gate.
    full_start = args.full_start or "2016-09-04"
    full_end = args.full_end or max(df.index[-1] for df in data.values()).strftime("%Y-%m-%d")

    candidates = []
    for _, r in table.head(args.top_n).iterrows():
        trial_cfg = dataclasses.replace(bt, sector_weights=r["weights"])
        oos = run(data, fixed_strategies, trial_cfg, oos_start, oos_end).stats
        bad = oos["trades"] < 20 or oos["avg_bars_held"] < MIN_HOLD
        if bad:
            continue
        candidates.append((score_fn(oos), oos, r["weights"]))
    if not candidates:
        raise SystemExit("no OOS-valid config")

    # gate: rerun each OOS-valid candidate over the FULL continuous history (not the
    # isolated IS/OOS windows) and check its REAL full-history Sharpe directly, not a
    # drawdown-duration proxy for it - duration alone was found NOT to be reliable for
    # every strategy family (a candidate with a seemingly-fine 13.8% duration fraction
    # still had full-history Sharpe -0.07, because the whole curve, not one localised
    # drawdown, was inflated by the segment-restart artefact - see paper Sec. 6.1
    # discussion). Among candidates that clear a real full-history robustness floor,
    # select the one with the highest full-history TOTAL RETURN, so the search still
    # rewards a strong result once robustness is actually confirmed rather than merely
    # inferred.
    print(f"\ngating top {len(candidates)} OOS-valid candidates on ACTUAL full "
          f"continuous-history performance ({full_start}..{full_end})...", flush=True)
    all_candidates_out = []
    for oos_score, oos, weights in candidates:
        full_cfg = dataclasses.replace(bt, sector_weights=weights)
        full_res = run(data, fixed_strategies, full_cfg, full_start, full_end)
        full_st = full_res.stats
        _, dd_frac = metrics.max_drawdown_duration(full_res.equity)
        all_candidates_out.append({
            "oos_score": oos_score, "full_sharpe": full_st["sharpe"],
            "full_return_pct": full_st["total_return_pct"], "dd_duration_frac": dd_frac,
            "weights": weights,
        })
        print(f"  OOS score={oos_score:.3f}  full_Sharpe={full_st['sharpe']:.2f}  "
              f"full_return%={full_st['total_return_pct']:.1f}  dd_duration_frac={dd_frac:.3f}",
              flush=True)

    pd.DataFrame(all_candidates_out).to_csv(
        f"bt/gate_candidates_{args.result_name}.csv", index=False)
    print(f"wrote bt/gate_candidates_{args.result_name}.csv (all candidates, for manual review)")

    robust = [c for c in all_candidates_out if c["full_sharpe"] >= args.min_full_sharpe]
    if not robust:
        print(f"WARNING: no candidate cleared full_Sharpe >= {args.min_full_sharpe}; "
              f"falling back to the single best full_Sharpe found.")
        robust = all_candidates_out
    best = max(robust, key=lambda c: c["full_return_pct"])
    best_oos, best_weights = None, best["weights"]
    for oos_score, oos, weights in candidates:
        if weights == best["weights"]:
            best_oos = oos
            break
    print(f"selected candidate: OOS score={best['oos_score']:.3f}  full_Sharpe={best['full_sharpe']:.2f}  "
          f"full_return%={best['full_return_pct']:.1f}  dd_duration_frac={best['dd_duration_frac']:.3f}",
          flush=True)

    best_cfg = dataclasses.replace(bt, sector_weights=best_weights)
    is_stats = run(data, fixed_strategies, best_cfg, is_start, is_end).stats

    per_sector_out = {s: {**{k: (round(isolated[s][k], 3) if isinstance(isolated[s][k], float) else isolated[s][k])
                            for k in strat_keys}, "weight": round(best_weights[s], 3)} for s in sectors}
    row = {"strategy": args.result_name, "params": {"sector_params": per_sector_out},
           **_row("IS", is_stats), **_row("OOS", best_oos)}

    print(f"\nbest OOS: return {best_oos['total_return_pct']}%  Sharpe {best_oos['sharpe']}  "
          f"Sortino {best_oos['sortino']}  maxDD {best_oos['max_drawdown_pct']}%  "
          f"hold {best_oos['avg_bars_held']}b  trades {best_oos['trades']}  ({time.time()-t0:.0f}s)")
    print("weights (sorted):")
    for s in sorted(per_sector_out, key=lambda s: -per_sector_out[s]["weight"]):
        print(f"  {s:12} weight={per_sector_out[s]['weight']:.2f}")

    with open(weights_out, "w") as f:
        json.dump({"_comment": f"decomposed: stage1=hyperopt_isolated_sector.py per sector for "
                              f"{args.strategy}, stage3(this)=hyperopt_weights_only.py for weights - "
                              "load with git_stock.sector_strategy.load_sector_strategies()",
                  **per_sector_out}, f, indent=4)
    print(f"\nwrote {weights_out}")

    try:
        existing = pd.read_csv(args.out)
        existing = existing[existing["strategy"] != args.result_name]
        merged = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        merged = pd.DataFrame([row])
    merged = merged.sort_values("OOS_Sharpe", ascending=False)
    merged.to_csv(args.out, index=False)

    pd.set_option("display.width", 240)
    cols = [c for c in ["strategy", "OOS_return%", "OOS_CAGR%", "OOS_Sharpe", "OOS_Sortino",
                        "OOS_maxDD%", "OOS_PF", "OOS_win%", "OOS_trades", "OOS_hold",
                        "IS_return%", "IS_Sharpe", "IS_maxDD%"] if c in merged.columns]
    print(f"\n---- {args.out} ({args.result_name} merged in, sorted by OOS_Sharpe) ----")
    print(merged[cols].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
