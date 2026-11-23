"""A deliberately "dumb" (no hyperopt, no ML, no weights) periodic universe check:
run the CURRENTLY LIVE strategy/params over a trailing window across the current
universe plus a curated pool of candidates that are NOT currently traded, then:

  - DITCH: a currently-included symbol with >= MIN_TRADES trades and a net loss
    over the trailing window -> drop it from its sector(s) in config.json.
  - ADD: a candidate symbol (not currently in any sector) with >= MIN_TRADES
    trades and a net profit over the trailing window -> add it to its assigned
    sector in config.json.

Nothing is weighted or hyperopted. The only action taken is editing config.json's
`sectors` dict (via the same _edit_sectors() helper the /add and /delete Telegram
commands use), so the change is exactly what a human would do by hand: add a
ticker line, or delete one. Default is a dry run that only prints what it WOULD
do; pass --apply to actually rewrite config.json.

    ./.venv/bin/python scripts/reevaluate_universe.py [--trailing-days 365] [--apply]
"""
from __future__ import annotations

import argparse

import pandas as pd

from git_stock.backtest import BacktestConfig, run
from git_stock.config import Config
from git_stock.data import DataFeed
from git_stock.strategies import get as get_strategy
from git_stock.telegrambot import _edit_sectors

MIN_TRADES = 3

# Curated pool of liquid, well-known tickers NOT currently in config.json, grouped
# under the SAME 19 sector names already used there. This is the only "judgement"
# baked into the script - which candidates are even worth test-driving. Anything
# that clears the bar gets added to the matching sector; nothing here is added
# automatically just for being on this list.
CANDIDATE_POOL = {
    "tech": ["ORCL", "ADBE", "AVGO", "QCOM", "IBM", "CSCO", "MU"],
    "health": ["CI", "HUM", "TMO", "DHR", "ZTS"],
    "goods": ["CLX", "HSY", "MDLZ", "EL"],
    "energy": ["MPC", "PSX", "EOG", "HAL"],
    "finance": ["BLK", "TFC", "USB", "PNC"],
    "industry": ["DE", "ITW", "EMR", "ETN"],
    "auto": ["RIVN", "LCID", "NIO"],
    "aerospace": ["TXT", "HEI", "TDG"],
    "civil": ["EME", "MAS", "VMC"],
    "tourism": ["UAL", "AAL", "NCLH", "TRIP"],
    "fashion": ["LVMUY", "UAA", "VFC"],
    "bio": ["ALNY", "BMRN", "EXAS"],
    "fintech": ["COIN", "SOFI", "UPST"],
    "comms": ["DIS", "WBD", "CHTR", "ERIC"],
    "realestate": ["O", "WELL", "AVB", "EXR"],
    "retail": ["KR", "BBY", "ORLY", "AZO"],
    "brazil": ["WEGE3.SA", "RENT3.SA", "LREN3.SA"],
    "asian": ["JD", "PDD"],
    "uk": ["RIO.L", "GSK.L", "AZN.L", "BATS.L"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trailing-days", type=int, default=730)
    ap.add_argument("--cache", default="stock_data")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--apply", action="store_true", help="actually edit config.json; default is dry-run")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    strat_name = cfg.strategy
    params = cfg.strategy_params or {}
    print(f"live strategy: {strat_name}  params: {params}")

    existing = set(cfg.tickers)
    candidates = {sym: sec for sec, syms in CANDIDATE_POOL.items() for sym in syms if sym not in existing}
    universe = sorted(existing | candidates.keys())
    sector_of = {t: sec for sec, ts in cfg.sectors.items() for t in ts}
    sector_of.update(candidates)  # candidate's assigned sector, for symbols not already mapped

    feed = DataFeed(args.cache, cfg.timeframe)
    data = feed.load_many(universe)
    missing_candidates = [s for s in candidates if s not in data]
    if missing_candidates:
        print(f"fetching {len(missing_candidates)} candidate(s) with no cached data yet: {missing_candidates}")
        feed.update(missing_candidates, cfg.daterange)
        data = feed.load_many(universe)
    missing = [s for s in universe if s not in data]
    if missing:
        stale_existing = [s for s in missing if s in existing]
        stale_candidates = [s for s in missing if s in candidates]
        if stale_existing:
            print(f"WARNING: {len(stale_existing)} EXISTING universe symbol(s) have no data at all "
                  f"(possibly delisted/renamed) - not touched by this script: {stale_existing}")
        if stale_candidates:
            print(f"no data available even after fetch attempt, skipping candidate(s): {stale_candidates}")
    data = {s: df for s, df in data.items() if df is not None and len(df)}

    last = max(df.index[-1] for df in data.values())
    start = (last - pd.Timedelta(days=args.trailing_days)).strftime("%Y-%m-%d")
    end = last.strftime("%Y-%m-%d")
    print(f"trailing window: {start} .. {end} ({args.trailing_days}d)  ·  "
          f"{len(existing & data.keys())} existing + {len(candidates.keys() & data.keys())} candidate symbols")

    bt = BacktestConfig(initial_balance=cfg.initial_balance, stake_amount=cfg.stake_amount,
        max_open_trades=cfg.max_open_trades, fee=0.003, slippage_pct=0.0005, spread_pct=0.0010,
        fill_next_open=True, gap_stops=True, timeframe=cfg.timeframe)

    strat = get_strategy(strat_name)(params)
    res = run(data, strat, bt, start, end)
    t = res.trades.copy()
    t["sector"] = t["symbol"].map(sector_of)

    by_sym = t.groupby("symbol").agg(pnl=("pnl", "sum"), trades=("pnl", "count"),
                                     win_rate=("pnl", lambda s: round((s > 0).mean() * 100, 1)))

    ditch = []
    for sym in sorted(existing):
        row = by_sym.loc[sym] if sym in by_sym.index else None
        if row is not None and row["trades"] >= MIN_TRADES and row["pnl"] < 0:
            ditch.append((sym, sector_of.get(sym, "?"), row["pnl"], int(row["trades"]), row["win_rate"]))

    add = []
    for sym, sec in sorted(candidates.items()):
        row = by_sym.loc[sym] if sym in by_sym.index else None
        if row is not None and row["trades"] >= MIN_TRADES and row["pnl"] > 0:
            add.append((sym, sec, row["pnl"], int(row["trades"]), row["win_rate"]))

    print(f"\n==== DITCH candidates (net loser, >= {MIN_TRADES} trades, trailing {args.trailing_days}d) ====")
    if ditch:
        for sym, sec, pnl, n, wr in sorted(ditch, key=lambda r: r[2]):
            print(f"  {sym:10} sector={sec:12} pnl={pnl:8.2f}  trades={n:3}  win%={wr}")
    else:
        print("  none")

    print(f"\n==== ADD candidates (net winner, >= {MIN_TRADES} trades, trailing {args.trailing_days}d) ====")
    if add:
        for sym, sec, pnl, n, wr in sorted(add, key=lambda r: -r[2]):
            print(f"  {sym:10} sector={sec:12} pnl={pnl:8.2f}  trades={n:3}  win%={wr}")
    else:
        print("  none")

    if not ditch and not add:
        print("\nno action needed.")
        return

    if not args.apply:
        print("\ndry run - nothing written. Re-run with --apply to make these changes to "
              f"{args.config}.")
        return

    def _mutate(sectors: dict) -> None:
        for sym, sec, *_ in ditch:
            for names in sectors.values():
                if sym in names:
                    names.remove(sym)
        for sym, sec, *_ in add:
            sectors.setdefault(sec, [])
            if sym not in sectors[sec]:
                sectors[sec].append(sym)

    _edit_sectors(args.config, _mutate)
    print(f"\napplied: removed {len(ditch)}, added {len(add)}  ->  {args.config} updated "
          f"(universe now {len(existing) - len(ditch) + len(add)} symbols)")


if __name__ == "__main__":
    main()
