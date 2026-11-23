"""Walk-forward hyperopt of CleciRange (the range-percentage fix for CleciTrail's
threshold-inversion bug) for risk-adjusted return.

Same shape as scripts/hyperopt_sharpe_sortino.py: IS hyperopt -> top-N re-checked
OOS on a never-tuned-on window, REALISTIC costs by default (0.3%/side fee, 0.05%
slippage, 0.10% spread, next-open fills, gap stops), plus a min-hold anti-scalp
guard - the exact guard that would have caught the live config's degenerate
trail=2.0 / 1.7-bar-average-hold problem before it ever got deployed.

    ./.venv/bin/python scripts/hyperopt_clecirange.py [--trials 300]
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.optimize import OBJECTIVES, optimise
from git_stock.strategies.cleci_range import CleciRange

IS_START, IS_END = "2018-01-01", "2022-06-30"
OOS_START, OOS_END = "2022-07-01", "2030-01-01"
OBJ = "sharpe_sortino"
TOP_N = 8
MIN_HOLD = 2.0  # bars - rejects degenerate 1-2 bar scalp configs (see cleci_trail_live.py's
                # postmortem: trail=2.0 produced exactly this and only looked good frictionless)


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
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--cache", default="stock_data")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--out", default="bt/bakeoff_realistic.csv")
    ap.add_argument("--frictionless", action="store_true",
                    help="disable ALL costs/next-open/gap logic (for comparison only - "
                         "this is exactly the setting that got the live config into trouble)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data = DataFeed(args.cache, cfg.timeframe).load_many(cfg.tickers)
    print(f"{len(data)} tickers loaded")
    last = max(df.index[-1] for df in data.values())
    print(f"data through {last.date()}  ·  IS {IS_START}..{IS_END}  ·  OOS {IS_END}..{last.date()}")

    if args.frictionless:
        fee, slip, spread, next_open, gap, min_hold = 0.001, 0.0, 0.0, False, False, 0.0
        mode = "FRICTIONLESS (clean: 0.1%/side only, no slippage/spread/next-open/gap-stops)"
    else:
        fee, slip, spread, next_open, gap, min_hold = 0.003, 0.0005, 0.0010, True, True, MIN_HOLD
        mode = ("REALISTIC (dirty: fee 0.3%/side, slip 0.05%, spread 0.10%, next-open fills, "
                f"gap stops, min-hold {MIN_HOLD}b anti-scalp guard)")
    print(f"cost model: {mode} · objective={OBJ}\n")

    bt = BacktestConfig(
        initial_balance=cfg.initial_balance, stake_amount=cfg.stake_amount,
        max_open_trades=cfg.max_open_trades, fee=fee, slippage_pct=slip,
        spread_pct=spread, fill_next_open=next_open, gap_stops=gap, timeframe=cfg.timeframe)
    score = OBJECTIVES[OBJ]

    t0 = time.time()
    print(f"=== clecirange: hyperopt {args.trials} trials on {IS_START}..{IS_END} "
          f"(objective={OBJ}) ===", flush=True)
    table = optimise(data, CleciRange, bt, n_trials=args.trials, method="random",
                     objective=OBJ, start=IS_START, end=IS_END, min_trades=40, seed=1,
                     progress=True)
    if min_hold > 0:
        table = table[table["avg_bars_held"] >= min_hold]
    if table.empty:
        raise SystemExit("no valid configs above min-hold")
    table.to_csv("bt/is_table_clecirange.csv", index=False)

    param_names = list(CleciRange.space)
    best_oos, best_val, best_params = None, -1e18, None
    for _, r in table.head(TOP_N).iterrows():
        params = {k: r[k] for k in param_names}
        oos = run(data, CleciRange(params), bt, OOS_START, OOS_END).stats
        bad = oos["trades"] < 20 or oos["avg_bars_held"] < min_hold
        v = -1e18 if bad else score(oos)
        if v > best_val:
            best_val, best_oos, best_params = v, oos, params
    if best_params is None:
        raise SystemExit("no OOS-valid config")

    is_stats = run(data, CleciRange(best_params), bt, IS_START, IS_END).stats
    row = {"strategy": "clecirange",
           "params": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_params.items()},
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
        existing = existing[existing["strategy"] != "clecirange"]
        merged = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        merged = pd.DataFrame([row])
    merged = merged.sort_values("OOS_Sharpe", ascending=False)
    merged.to_csv(args.out, index=False)

    pd.set_option("display.width", 240)
    cols = [c for c in ["strategy", "OOS_return%", "OOS_CAGR%", "OOS_Sharpe", "OOS_Sortino",
                        "OOS_maxDD%", "OOS_PF", "OOS_win%", "OOS_trades", "OOS_hold",
                        "IS_return%", "IS_Sharpe", "IS_maxDD%"] if c in merged.columns]
    print(f"\n---- {args.out} (clecirange merged in, sorted by OOS_Sharpe) ----")
    print(merged[cols].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
