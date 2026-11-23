"""Periodic retraining pipeline for a decomposed (L4) sector strategy.

Design rationale (from the per-sector-optimization study): short windows are
dominated by noise and single shared market events (see the 1y/2y comparison
tables - rankings reshuffle completely year to year), so this pipeline does NOT
retrain on a short recent window. Instead it:

  - fits stage 1 (per-sector params) and stage 3 (weights) on an IS window that
    is everything available EXCLUDING the most recent `--oos-years` (default 2),
    so the OOS check is genuinely held out and reasonably sized
  - gates every OOS-valid candidate on its REAL full-continuous-history Sharpe,
    exactly like the corrective gate already built for the main study
    (hyperopt_weights_only.py) - not a duration proxy, which was found
    unreliable for some strategy families
  - re-evaluates the CURRENTLY DEPLOYED (incumbent) config fresh, over that same
    new full-history window, and only recommends promotion if the challenger
    clears the full-Sharpe floor AND beats the incumbent's OWN freshly-measured
    full-history return by a margin (avoids churning on noise between two
    candidates that are actually indistinguishable)

This script never overwrites the live config/sector_params_*.json - it writes
the challenger to a side file and prints a recommendation. Promotion is a
deliberate, separate, human step (`cp` the candidate file over the live one)
since this eventually feeds a live-trading bot.

    ./.venv/bin/python scripts/retrain_pipeline.py --strategy clecirangecross \\
        --result-name clecirangecrossdecomposedsectorgated [--as-of 2026-10-03] \\
        [--trials-isolated 2000] [--trials-weights 3000] [--top-n 300]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys

import pandas as pd

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.strategies import get as get_strategy

OOS_YEARS = 2
IS_YEARS = 8                 # IS window length immediately before the OOS slice
COVERAGE_FLOOR = 0.85        # full_start = earliest date >= this fraction of tickers already cover
MIN_FULL_SHARPE = 0.3
PROMOTE_MARGIN = 0.05        # challenger must beat incumbent's full Sharpe by this much to recommend promotion


def _resolve_full_start(data: dict[str, pd.DataFrame], coverage: float) -> str:
    """Earliest date at which at least `coverage` fraction of the universe already has
    data - generalises the study's fixed 2016-09-04 anchor (chosen by hand for the old,
    pre-backfill cache) so it keeps making sense as the cache is extended further back."""
    starts = sorted(df.index[0] for df in data.values())
    idx = min(int(len(starts) * coverage), len(starts) - 1)
    return starts[idx].strftime("%Y-%m-%d")


def _run_stage(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, help="registered base strategy, e.g. clecirangecross")
    ap.add_argument("--result-name", required=True,
                    help="incumbent's strategy label in bt/bakeoff_realistic.csv, "
                         "e.g. clecirangecrossdecomposedsectorgated")
    ap.add_argument("--as-of", default=None, help="retrain as of this date (default: latest cached bar)")
    ap.add_argument("--trials-isolated", type=int, default=2000)
    ap.add_argument("--trials-weights", type=int, default=3000)
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--min-full-sharpe", type=float, default=MIN_FULL_SHARPE)
    ap.add_argument("--cache", default="stock_data")
    ap.add_argument("--config", default="config/config.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data = DataFeed(args.cache, cfg.timeframe).load_many(cfg.tickers)
    sector_of = {t: sec for sec, ts in cfg.sectors.items() for t in ts}

    as_of = pd.Timestamp(args.as_of) if args.as_of else max(df.index[-1] for df in data.values())
    oos_end = as_of
    oos_start = as_of - pd.DateOffset(years=OOS_YEARS)
    is_end = oos_start
    is_start = as_of - pd.DateOffset(years=IS_YEARS)
    full_start = _resolve_full_start(data, COVERAGE_FLOOR)
    full_end = as_of.strftime("%Y-%m-%d")
    oos_start, oos_end, is_start, is_end = (d.strftime("%Y-%m-%d") for d in (oos_start, oos_end, is_start, is_end))

    print(f"=== retrain {args.strategy} as of {full_end} ===")
    print(f"  IS   window: {is_start} .. {is_end}  ({IS_YEARS}y)")
    print(f"  OOS  window: {oos_start} .. {oos_end}  ({OOS_YEARS}y, held out)")
    print(f"  FULL window: {full_start} .. {full_end}  ({COVERAGE_FLOOR:.0%} coverage floor)")

    candidate_tag = f"{args.result_name}_candidate_{full_end}"
    isolated_glob = f"bt/isolated_{args.strategy}_retrain_*.csv"

    sectors = sorted(cfg.sectors)
    for sec in sectors:
        _run_stage([
            sys.executable, "scripts/hyperopt_isolated_sector.py",
            "--strategy", args.strategy, "--sector", sec,
            "--trials", str(args.trials_isolated),
            "--cache", args.cache, "--config", args.config,
            "--out", f"bt/isolated_{args.strategy}_retrain_{sec}.csv",
            "--is-start", is_start, "--is-end", is_end,
            "--oos-start", oos_start, "--oos-end", oos_end,
        ])

    bakeoff_scratch = f"bt/bakeoff_retrain_{args.strategy}.csv"
    weights_out = f"config/sector_params_{args.strategy}_{candidate_tag}.json"
    _run_stage([
        sys.executable, "scripts/hyperopt_weights_only.py",
        "--strategy", args.strategy, "--result-name", candidate_tag,
        "--trials", str(args.trials_weights), "--top-n", str(args.top_n),
        "--min-full-sharpe", str(args.min_full_sharpe),
        "--cache", args.cache, "--config", args.config,
        "--out", bakeoff_scratch, "--isolated-glob", isolated_glob, "--weights-out", weights_out,
        "--is-start", is_start, "--is-end", is_end,
        "--oos-start", oos_start, "--oos-end", oos_end,
        "--full-start", full_start, "--full-end", full_end,
    ])

    challenger_row = pd.read_csv(bakeoff_scratch)
    challenger_row = challenger_row[challenger_row["strategy"] == candidate_tag].iloc[0]

    # re-evaluate the challenger's own full-history numbers directly (bakeoff only stores
    # IS/OOS) and the incumbent's, fresh, over the identical window - apples to apples,
    # since the incumbent was itself only ever measured against an earlier as_of.
    cls = get_strategy(args.strategy)

    def _full_stats(sector_params: dict) -> dict:
        strat = {s: cls({k: v for k, v in p.items() if k != "weight"}) for s, p in sector_params.items()}
        weights = {s: p["weight"] for s, p in sector_params.items()}
        bt = BacktestConfig(initial_balance=cfg.initial_balance, stake_amount=170.0, max_open_trades=30,
            fee=0.003, slippage_pct=0.0005, spread_pct=0.0010, fill_next_open=True, gap_stops=True,
            timeframe=cfg.timeframe, sector_of=sector_of, sector_weights=weights)
        return run(data, strat, bt, full_start, full_end).stats

    with open(weights_out) as f:
        challenger_params = {k: v for k, v in json.load(f).items() if k != "_comment"}
    challenger_full = _full_stats(challenger_params)

    incumbent_path = f"config/sector_params_{args.strategy}_decomposed.json"
    try:
        with open(incumbent_path) as f:
            incumbent_params = {k: v for k, v in json.load(f).items() if k != "_comment"}
        incumbent_full = _full_stats(incumbent_params)
        incumbent_label = incumbent_path
    except FileNotFoundError:
        incumbent_full = None
        incumbent_label = "(none deployed yet)"

    print(f"\n=== retrain report: {args.strategy} ===")
    print(f"incumbent: {incumbent_label}")
    if incumbent_full is not None:
        print(f"  full-history [{full_start}..{full_end}]: Sharpe={incumbent_full['sharpe']:.2f}  "
              f"return%={incumbent_full['total_return_pct']:.1f}  maxDD%={incumbent_full['max_drawdown_pct']:.1f}")
    print(f"challenger: {weights_out}")
    print(f"  OOS [{oos_start}..{oos_end}]: Sharpe={challenger_row['OOS_Sharpe']}  "
          f"return%={challenger_row['OOS_return%']}")
    print(f"  full-history [{full_start}..{full_end}]: Sharpe={challenger_full['sharpe']:.2f}  "
          f"return%={challenger_full['total_return_pct']:.1f}  maxDD%={challenger_full['max_drawdown_pct']:.1f}")

    clears_floor = challenger_full["sharpe"] >= args.min_full_sharpe
    beats_incumbent = incumbent_full is None or challenger_full["sharpe"] >= incumbent_full["sharpe"] + PROMOTE_MARGIN
    recommend = clears_floor and beats_incumbent
    print(f"\nrecommendation: {'PROMOTE challenger' if recommend else 'KEEP incumbent'}")
    if not clears_floor:
        print(f"  (challenger full-Sharpe {challenger_full['sharpe']:.2f} below floor {args.min_full_sharpe})")
    if incumbent_full is not None and not beats_incumbent:
        print(f"  (challenger full-Sharpe {challenger_full['sharpe']:.2f} does not clear incumbent "
              f"{incumbent_full['sharpe']:.2f} + margin {PROMOTE_MARGIN})")
    print(f"\nThis script never overwrites {incumbent_path}. To promote the challenger:")
    print(f"  cp {weights_out} {incumbent_path}")


if __name__ == "__main__":
    main()
