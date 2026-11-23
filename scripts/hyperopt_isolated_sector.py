"""Stage 1 of the decomposed per-sector search, generalised to any registered
strategy: hyperopt the strategy's own params for ONE sector IN ISOLATION - only
that sector's own tickers, no competition from the other 18 sectors for capital
slots (max_open_trades is set to the sector's own ticker count so essentially
every signal gets taken; this stage measures the sector's raw pattern quality,
not portfolio-level capital contention).

Different from a joint per-sector search (e.g. hyperopt_swingtrend_full_sector.py),
which searches all sectors' params AND weights together in one big space scored
by the full portfolio at once. This is the alternative, lower-dimensional path:
N independent small searches, each against a much smaller dataset - then a
separate weight-only stage (hyperopt_weights_only.py) assembles them and tunes
stake sizing on the full portfolio.

    ./.venv/bin/python scripts/hyperopt_isolated_sector.py --strategy bbrsituned --sector tech [--trials 2000]
"""
from __future__ import annotations

import argparse
import random
import time

import pandas as pd

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.optimize import OBJECTIVES, _sample
from git_stock.strategies import get as get_strategy

IS_START, IS_END = "2018-01-01", "2022-06-30"
OOS_START, OOS_END = "2022-07-01", "2030-01-01"
OBJ = "sharpe_sortino"
TOP_N = 15
MIN_HOLD = 2.0
MIN_TRADES = 15   # lower than the 40 used for full-universe searches - a single
                  # sector has far fewer tickers, so far fewer possible trades


def _row(tag, st):
    return {
        f"{tag}_return%": st["total_return_pct"], f"{tag}_Sharpe": st["sharpe"],
        f"{tag}_Sortino": st["sortino"], f"{tag}_maxDD%": st["max_drawdown_pct"],
        f"{tag}_trades": st["trades"], f"{tag}_hold": st["avg_bars_held"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, help="registered strategy name, e.g. swingtrendtuned")
    ap.add_argument("--sector", required=True)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--cache", default="stock_data")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--is-start", default=None, help="override IS_START (default: module constant)")
    ap.add_argument("--is-end", default=None, help="override IS_END (default: module constant)")
    ap.add_argument("--oos-start", default=None, help="override OOS_START (default: module constant)")
    ap.add_argument("--oos-end", default=None, help="override OOS_END (default: module constant)")
    args = ap.parse_args()
    is_start, is_end = args.is_start or IS_START, args.is_end or IS_END
    oos_start, oos_end = args.oos_start or OOS_START, args.oos_end or OOS_END
    out_path = args.out or f"bt/isolated_{args.strategy}_{args.sector}.csv"
    cls = get_strategy(args.strategy)
    space = cls.space

    cfg = Config.load(args.config)
    tickers = cfg.sectors[args.sector]
    data = DataFeed(args.cache, cfg.timeframe).load_many(tickers)
    if len(data) < 2:
        print(f"sector={args.sector}: only {len(data)} ticker(s) with data, skipping")
        return

    bt = BacktestConfig(initial_balance=cfg.initial_balance, stake_amount=170.0, max_open_trades=len(data),
        fee=0.003, slippage_pct=0.0005, spread_pct=0.0010, fill_next_open=True, gap_stops=True,
        timeframe=cfg.timeframe)
    score_fn = OBJECTIVES[OBJ]
    rng = random.Random(1)

    t0 = time.time()
    print(f"=== isolated {args.strategy} sector={args.sector} ({len(data)} tickers): "
          f"hyperopt {args.trials} trials on {is_start}..{is_end} ===", flush=True)
    rows = []
    for i in range(args.trials):
        params = _sample(space, rng)
        try:
            res = run(data, cls(params), bt, is_start, is_end)
        except Exception:
            continue
        st = res.stats
        penalised = st["trades"] < MIN_TRADES
        rows.append({**params, **st, "score": (-9e9 if penalised else score_fn(st))})

    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    table = table[table["avg_bars_held"] >= MIN_HOLD]
    if table.empty:
        print(f"sector={args.sector}: no valid configs above min-hold")
        return

    best_oos, best_val, best_params = None, -1e18, None
    for _, r in table.head(TOP_N).iterrows():
        params = {k: r[k] for k in space}
        try:
            oos = run(data, cls(params), bt, oos_start, oos_end).stats
        except Exception:
            continue
        bad = oos["trades"] < 8 or oos["avg_bars_held"] < MIN_HOLD
        v = -1e18 if bad else score_fn(oos)
        if v > best_val:
            best_val, best_oos, best_params = v, oos, params
    if best_params is None:
        print(f"sector={args.sector}: no OOS-valid config, falling back to best-IS")
        best_params = {k: table.iloc[0][k] for k in space}
        best_oos = run(data, cls(best_params), bt, oos_start, oos_end).stats

    is_stats = run(data, cls(best_params), bt, is_start, is_end).stats
    row = {"sector": args.sector, "n_tickers": len(data),
           **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_params.items()},
           **_row("IS", is_stats), **_row("OOS", best_oos)}

    print(f"sector={args.sector}  best OOS: Sharpe {best_oos['sharpe']}  Sortino {best_oos['sortino']}  "
          f"return {best_oos['total_return_pct']}%  trades {best_oos['trades']}  params={best_params}  "
          f"({time.time()-t0:.0f}s)")
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
