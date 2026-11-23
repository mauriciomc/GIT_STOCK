"""Command-line interface:  python -m git_stock <command> [options]"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import pandas as pd

from . import strategies
from .backtest import BacktestConfig, run
from .config import Config
from .data import DataFeed
from .optimize import optimise
from .signals import SignalRunner


def _timerange(s: str | None):
    if not s:
        return None, None
    a, _, b = s.partition("-")
    fmt = lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if x else None
    return fmt(a), fmt(b)


def _bt_config(cfg: Config) -> BacktestConfig:
    return BacktestConfig(
        initial_balance=cfg.initial_balance, stake_amount=cfg.stake_amount,
        max_open_trades=cfg.max_open_trades, fee=cfg.open_fee,
        stoploss_pct=cfg.stop_loss, take_profit_pct=cfg.take_profit,
        timeframe=cfg.timeframe)


def cmd_strategies(args, cfg):
    for name in strategies.available():
        s = strategies.get(name)
        print(f"  {name:12} params: {list(s.space)}")


def cmd_data(args, cfg):
    DataFeed(args.cache, cfg.timeframe).update(cfg.tickers, cfg.daterange)


def cmd_repair(args, cfg):
    ok, bad = DataFeed(args.cache, cfg.timeframe).repair(cfg.tickers)
    print(f"repaired {ok} cached CSVs to canonical schema · {bad} unusable")


def cmd_backtest(args, cfg):
    start, end = _timerange(args.timerange)
    data = DataFeed(args.cache, cfg.timeframe).load_many(cfg.tickers)
    if not data:
        sys.exit("no cached data — run `python -m git_stock data` first")
    strat_cls = strategies.get(args.strategy or cfg.strategy)
    p0 = {"window": cfg.timewindow} if "window" in strat_cls.space else {}
    if (args.strategy or cfg.strategy).lower() == cfg.strategy.lower():
        p0.update(cfg.strategy_params or {})
    strat = strat_cls(p0)
    res = run(data, strat, _bt_config(cfg), start, end)
    print(f"\n{strat.name}  {res.params}")
    print(res.summary())
    for k, v in res.stats.items():
        print(f"  {k:22} {v}")
    if len(res.trades):
        by_sym = (res.trades.groupby("symbol")["pnl"].agg(["count", "sum"])
                  .sort_values("sum", ascending=False))
        print("\ntop / bottom symbols by P/L:")
        print(pd.concat([by_sym.head(5), by_sym.tail(5)]).to_string())
    if args.out:
        p = pathlib.Path(args.out)
        p.mkdir(parents=True, exist_ok=True)
        res.trades.to_csv(p / "trades.csv", index=False)
        res.equity.to_csv(p / "equity.csv")
        print(f"\nwrote {p}/trades.csv, equity.csv")


def cmd_optimise(args, cfg):
    start, end = _timerange(args.timerange)
    data = DataFeed(args.cache, cfg.timeframe).load_many(cfg.tickers)
    if not data:
        sys.exit("no cached data — run `python -m git_stock data` first")
    strat_cls = strategies.get(args.strategy or cfg.strategy)
    df = optimise(data, strat_cls, _bt_config(cfg), n_trials=args.trials,
                  method=args.method, objective=args.objective,
                  start=start, end=end, min_trades=args.min_trades, seed=args.seed)
    cols = list(strat_cls.space) + ["score", "total_return_pct", "cagr_pct", "sharpe",
                                    "max_drawdown_pct", "win_rate_pct", "trades"]
    print(f"\ntop {min(args.top, len(df))} by {args.objective}:")
    print(df[cols].head(args.top).to_string(index=False))
    print("\nbest params:", df.iloc[0][list(strat_cls.space)].to_dict())
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


def _strip_html(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", s)


def cmd_signals(args, cfg):
    sr = SignalRunner(cfg, db_path=f"databases/{cfg.database}", cache_dir=args.cache)
    r = sr.run(update=not args.no_update)
    print(_strip_html(r.as_text()))

    tg = cfg.telegram or {}
    if not (tg.get("enabled") and tg.get("token") and not args.quiet):
        return
    from .charts import plot_signals
    from .notify import send_message, send_photo
    token, chat = tg["token"], tg.get("chat_id", "")
    send_message(token, chat, r.as_text(), buttons=r.buttons() or None)
    if tg.get("send_charts") and (r.opened or r.closed):
        data = sr.feed.load_many([t["symbol"] for t in r.opened + r.closed])
        for sym, df in data.items():
            send_photo(token, chat,
                      plot_signals(sr.strategy.prepare(df), sym, sr.strategy.name,
                                  real_trades=sr.trades_for(sym)),
                      f"<b>{sym}</b> — {sr.strategy.name}")


def cmd_seed(args, cfg):
    from .seed import positions_from_yaml, seed_ledger
    allpos = positions_from_yaml(args.from_yaml)
    pos = allpos if args.all else [p for p in allpos if p["priceable"]]
    excluded = [p for p in allpos if p not in pos]
    print(f"stock/etf/fund positions in {args.from_yaml}: {len(allpos)} found, "
          f"{len(pos)} to seed{' (all)' if args.all else ' (priceable only; --all for the rest)'}")
    rows = [[p["symbol"], p["asset_class"], p["account"], f"£{p['value_base']:,.0f}",
             "live" if p["priceable"] else "static"] for p in pos]
    print(pd.DataFrame(rows, columns=["symbol", "class", "account", "value", "price"]).to_string(index=False))
    print(f"total to seed: £{sum(p['value_base'] for p in pos):,.0f}")
    if excluded:
        print(f"\nexcluded (no Yahoo price — pass --all to include as static positions): "
              f"{', '.join(p['symbol'] for p in excluded)}")
    if args.dry_run:
        print("\n--dry-run: nothing written"); return
    sr = SignalRunner(cfg, db_path=f"databases/{cfg.database}", cache_dir=args.cache)
    done = seed_ledger(sr, pos, force=args.force)
    print(f"\nseeded {len(done)} open paper positions into databases/{cfg.database} "
          f"(tag 'seed'). One-off copy — the bot and the dashboard are not linked.")


def cmd_telegram_bot(args, cfg):
    from .telegrambot import run_bot
    run_bot(cfg, cache=args.cache, config_path=args.config)


def cmd_report(args, cfg):
    sr = SignalRunner(cfg, db_path=f"databases/{cfg.database}", cache_dir=args.cache)
    op, cl = sr.open_trades(), sr.closed_trades()
    print(f"open positions: {len(op)}")
    if len(op):
        print(op[["symbol", "open_date", "open_price", "stake"]].to_string(index=False))
    print(f"\nclosed trades: {len(cl)}   realised P/L: {cl['pnl'].sum():.2f}" if len(cl) else "\nno closed trades")
    if len(cl):
        print(cl[["symbol", "open_date", "close_date", "open_price", "close_price", "pnl"]].tail(15).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("git_stock", description="stock signal / backtest / optimise toolkit")
    p.add_argument("-c", "--config", default="config/config.json")
    p.add_argument("--cache", default="data_cache")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("strategies", help="list strategies").set_defaults(func=cmd_strategies)
    sub.add_parser("data", help="fetch / update the price cache").set_defaults(func=cmd_data)
    sub.add_parser("repair", help="rewrite cached CSVs in canonical schema").set_defaults(func=cmd_repair)

    b = sub.add_parser("backtest", help="run a backtest")
    b.add_argument("-s", "--strategy")
    b.add_argument("-t", "--timerange", help="YYYYMMDD-YYYYMMDD")
    b.add_argument("-o", "--out", help="dir to write trades.csv / equity.csv")
    b.set_defaults(func=cmd_backtest)

    o = sub.add_parser("optimise", help="hyperopt-style parameter search")
    o.add_argument("-s", "--strategy")
    o.add_argument("-t", "--timerange")
    o.add_argument("-n", "--trials", type=int, default=200)
    o.add_argument("--method", choices=["random", "grid", "optuna"], default="random")
    o.add_argument("--objective", choices=["sharpe", "sortino", "calmar", "return", "expectancy", "recipe"], default="sharpe")
    o.add_argument("--min-trades", type=int, default=10)
    o.add_argument("--top", type=int, default=10)
    o.add_argument("--seed", type=int, default=42)
    o.add_argument("-o", "--out", help="csv path for the full results table")
    o.set_defaults(func=cmd_optimise)

    s = sub.add_parser("signals", help="run the daily paper-signal runner")
    s.add_argument("--no-update", action="store_true", help="skip the data refresh")
    s.add_argument("--quiet", action="store_true", help="don't send Telegram")
    s.set_defaults(func=cmd_signals)

    sd = sub.add_parser("seed", help="one-off import of a starting book from a dashboard portfolio.yaml")
    sd.add_argument("--from", dest="from_yaml", required=True, metavar="PORTFOLIO_YAML")
    sd.add_argument("--all", action="store_true", help="also seed positions with no Yahoo price (e.g. pension funds) as static holdings")
    sd.add_argument("--dry-run", action="store_true")
    sd.add_argument("--force", action="store_true", help="reseed even if the ledger has open trades")
    sd.set_defaults(func=cmd_seed)

    tb = sub.add_parser("telegram-bot", help="run the Telegram command / chart-button bot")
    tb.set_defaults(func=cmd_telegram_bot)

    sub.add_parser("report", help="show the paper portfolio").set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load(args.config)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
