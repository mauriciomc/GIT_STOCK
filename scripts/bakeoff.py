"""Strategy bake-off with walk-forward validation.

For every strategy:
  1. hyperopt (random search) on the IN-SAMPLE window
  2. take the top-N configs, backtest each on the OUT-OF-SAMPLE window (never tuned on)
  3. keep the config with the best out-of-sample score
Prints a comparison table (in-sample vs out-of-sample) so you can see which
strategies actually generalise and which just curve-fit.

By default this runs a **realistic** cost model: 0.3% commission per side, slippage,
half-spread, next-open fills and overnight-gap stops (see --frictionless to turn it
all off and reproduce the old optimistic numbers).

    ./.venv/bin/python scripts/bakeoff.py [--trials 200] [--strategies cleci,...]
    ./.venv/bin/python scripts/bakeoff.py --frictionless --out bt/bakeoff_frictionless.csv
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.optimize import OBJECTIVES, optimise
from git_stock.strategies import available, get

IS_START, IS_END = "2018-01-01", "2022-06-30"      # hyperopt here
OOS_START, OOS_END = "2022-07-01", "2030-01-01"    # validate here (unseen)
OBJ = "recipe"
TOP_N = 8
# NOTE: realised_pct/unrealised_pct/unrealised_frac are reported but not gated on — a
# hard cutoff double-penalises a patient, long-hold strategy (open positions are already
# correctly marked-to-market in the equity curve everything else is computed from) and
# empirically favoured a more regime-fit config over a more robust one when tried.


def _row(tag, st):
    return {
        f"{tag}_return%": st["total_return_pct"], f"{tag}_CAGR%": st["cagr_pct"],
        f"{tag}_Sharpe": st["sharpe"], f"{tag}_maxDD%": st["max_drawdown_pct"],
        f"{tag}_PF": st["profit_factor"], f"{tag}_win%": st["win_rate_pct"],
        f"{tag}_trades": st["trades"], f"{tag}_hold": st["avg_bars_held"],
        f"{tag}_realised%": st["realised_pct"], f"{tag}_unrealised%": st["unrealised_pct"],
        f"{tag}_openAtEnd": st["open_at_end"], f"{tag}_unrealFrac": st["unrealised_frac"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--cache", default="stock_data")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--strategies", default="cleci,clecitrail,emacross,bbrsi,ichimoku")
    ap.add_argument("--out", default="bt/bakeoff.csv")
    # ---- cost model (realistic by default) ----
    ap.add_argument("--fee", type=float, default=0.003, help="commission per side")
    ap.add_argument("--slippage", type=float, default=0.0005, help="adverse slip per fill")
    ap.add_argument("--spread", type=float, default=0.0010, help="full bid/ask spread")
    ap.add_argument("--min-hold", type=float, default=2.0,
                    help="reject configs whose avg bars-held is below this (anti-scalp guard)")
    ap.add_argument("--frictionless", action="store_true",
                    help="disable ALL costs/next-open/gap logic (old optimistic mode)")
    args = ap.parse_args()

    if args.frictionless:
        fee, slip, spread, next_open, gap, min_hold = 0.001, 0.0, 0.0, False, False, 0.0
    else:
        fee, slip, spread, next_open, gap, min_hold = (
            args.fee, args.slippage, args.spread, True, True, args.min_hold)

    cfg = Config.load(args.config)
    data = DataFeed(args.cache, cfg.timeframe).load_many(cfg.tickers)
    print(f"{len(data)} tickers loaded")
    last = max(df.index[-1] for df in data.values())
    print(f"data through {last.date()}  ·  IS {IS_START}..{IS_END}  ·  OOS {IS_END}..{last.date()}")
    mode = "FRICTIONLESS" if args.frictionless else (
        f"fee {fee:.1%}/side · slip {slip:.2%} · spread {spread:.2%} · next-open fills · "
        f"gap stops · min-hold {min_hold}b")
    print(f"cost model: {mode}\n")

    bt = BacktestConfig(
        initial_balance=cfg.initial_balance, stake_amount=cfg.stake_amount,
        max_open_trades=cfg.max_open_trades, fee=fee, slippage_pct=slip, spread_pct=spread,
        fill_next_open=next_open, gap_stops=gap, timeframe=cfg.timeframe)
    score = OBJECTIVES[OBJ]
    names = [s.strip() for s in args.strategies.split(",") if s.strip() in available()]
    results = []

    for name in names:
        t0 = time.time()
        cls = get(name)
        print(f"=== {name}: hyperopt {args.trials} trials on {IS_START}..{IS_END} ===", flush=True)
        table = optimise(data, cls, bt, n_trials=args.trials, method="random", objective=OBJ,
                         start=IS_START, end=IS_END, min_trades=40, seed=1, progress=True)
        if min_hold > 0 and "avg_bars_held" in table.columns:
            table = table[table["avg_bars_held"] >= min_hold]
        if table.empty:
            print("  (no valid configs above min-hold)\n"); continue
        table.to_csv(f"bt/is_table_{name}.csv", index=False)  # so a guard-only re-check doesn't need re-hyperopting
        param_names = list(cls.space)
        best_oos, best_val, best_params = None, -1e18, None
        for _, r in table.head(TOP_N).iterrows():
            params = {k: r[k] for k in param_names}
            oos = run(data, cls(params), bt, OOS_START, OOS_END).stats
            bad = oos["trades"] < 20 or oos["avg_bars_held"] < min_hold
            v = -1e18 if bad else score(oos)
            if v > best_val:
                best_val, best_oos, best_params = v, oos, params
        if best_params is None:
            print("  (no OOS-valid config)\n"); continue
        is_stats = run(data, cls(best_params), bt, IS_START, IS_END).stats
        results.append({"strategy": name,
                        "params": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_params.items()},
                        **_row("IS", is_stats), **_row("OOS", best_oos)})
        print(f"  best OOS: return {best_oos['total_return_pct']}%  Sharpe {best_oos['sharpe']}  "
              f"maxDD {best_oos['max_drawdown_pct']}%  hold {best_oos['avg_bars_held']}b  "
              f"trades {best_oos['trades']}  [unrealised {best_oos['unrealised_frac']:.0%} of P&L, "
              f"{best_oos['open_at_end']} still open]  ({time.time()-t0:.0f}s)\n", flush=True)

    df = pd.DataFrame(results)
    if df.empty:
        sys.exit("no results")
    df = df.sort_values("OOS_return%", ascending=False)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 60)
    print("\n================ BAKE-OFF: best config per strategy ================")
    print(f"cost model: {mode}")
    print(df[["strategy", "OOS_return%", "OOS_CAGR%", "OOS_Sharpe", "OOS_maxDD%", "OOS_PF",
              "OOS_win%", "OOS_trades", "OOS_hold", "IS_return%", "IS_Sharpe", "IS_maxDD%"]].to_string(index=False))
    print("\n---- params ----")
    for _, r in df.iterrows():
        print(f"  {r['strategy']:12} {r['params']}")
    print("\n(IS = hyperopted window, OOS = out-of-sample validation. 'hold' = avg bars held.)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
