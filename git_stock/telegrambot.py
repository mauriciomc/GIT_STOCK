"""Lightweight Telegram polling bot — answers commands and chart buttons.

Run it alongside the daily `signals` job:  python -m git_stock telegram-bot
Commands:  /status   /report   /chart SYMBOL   /add SYMBOL SECTOR   /delete SYMBOL [SECTOR]   /help
Inline buttons on signal messages ("📈 AAPL") call back here to send the chart.
"""
from __future__ import annotations

import json
import logging
import pathlib
import threading
import time

from .config import Config
from .data import DataFeed
from .notify import answer_callback, get_bot_username, get_updates, mono_table, send_message, send_photo, set_my_commands
from .signals import SignalRunner
from .strategies import get as get_strategy

log = logging.getLogger("git_stock.telegrambot")

# populates the "/" menu button in Telegram clients (the same feature freqtrade's
# bot uses). /help below has its own richer per-command usage text; keep the two
# in sync by command NAME (every command here should also appear in /help).
_COMMANDS: list[tuple[str, str]] = [
    ("status", "Portfolio summary + open positions"),
    ("strategy", "Full strategy config and params"),
    ("report", "Recently closed trades"),
    ("symbols", "List the trading universe (optionally one sector)"),
    ("chart", "Signal chart for a symbol: /chart SYMBOL [bars]"),
    ("add", "Add a ticker: /add SYMBOL SECTOR"),
    ("delete", "Remove a ticker: /delete SYMBOL [SECTOR]"),
    ("help", "Show this command list"),
]


def _edit_sectors(config_path: str, mutate) -> dict:
    """Read config.json fresh, hand its `sectors` dict to `mutate` (in place), write the
    WHOLE raw JSON back unchanged apart from that - never round-trips through the Config
    dataclass, so fields it doesn't model (_strategy_note, etc.) survive untouched."""
    p = pathlib.Path(config_path)
    raw = json.loads(p.read_text())
    raw.setdefault("sectors", {})
    mutate(raw["sectors"])
    p.write_text(json.dumps(raw, indent=4))
    return raw["sectors"]


def _chart(cfg: Config, cache: str, symbol: str, lookback: int = 400) -> bytes | None:
    try:
        from .charts import plot_signals
    except ImportError:
        log.warning("matplotlib not installed — charts disabled")
        return None
    feed = DataFeed(cache, cfg.timeframe)
    df = feed.load(symbol)
    if df is None:
        feed.update([symbol], cfg.daterange)
        df = feed.load(symbol)
    if df is None:
        return None
    strat = get_strategy(cfg.strategy)({"window": cfg.timewindow, **(cfg.strategy_params or {})})
    sr = SignalRunner(cfg, db_path=f"databases/{cfg.database}", cache_dir=cache)
    return plot_signals(strat.prepare(df), symbol, strat.describe(), lookback=lookback,
                        real_trades=sr.trades_for(symbol))


def _strategy_line(cfg: Config) -> str:
    strat = get_strategy(cfg.strategy)({"window": cfg.timewindow, **(cfg.strategy_params or {})})
    return strat.describe()


# ---- live prices for open positions, rate-limited to one fetch per 15 min per symbol ----
# Only ever called with the (small) set of currently-open symbols, never the full universe
# — so this stays cheap regardless of how often /status is polled.
_QUOTE_TTL = 15 * 60
_quote_cache: dict[str, tuple[float, float]] = {}   # symbol -> (last price, fetched_at epoch)


def _live_prices(symbols: list[str], feed: DataFeed) -> dict[str, float | None]:
    """Current price per symbol: a live 15-minute intraday quote if we have (or can fetch)
    one no older than 15 minutes, else the last cached daily close, else None."""
    now = time.time()
    stale = [s for s in dict.fromkeys(symbols) if now - _quote_cache.get(s, (0.0, 0.0))[1] > _QUOTE_TTL]
    if stale:
        try:
            import yfinance as yf
            raw = yf.download(stale, period="1d", interval="15m", group_by="ticker",
                              auto_adjust=False, progress=False, threads=False)
            for s in stale:
                try:
                    col = raw[s] if len(stale) > 1 else raw
                    closes = col["Close"].dropna()
                    if len(closes):
                        _quote_cache[s] = (float(closes.iloc[-1]), now)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
            log.warning("live quote fetch failed for %s: %s", stale, e)

    out: dict[str, float | None] = {}
    for s in symbols:
        if s in _quote_cache:
            out[s] = _quote_cache[s][0]
        else:
            df = feed.load(s)                       # fallback: previous day's close
            out[s] = float(df["close"].iloc[-1]) if df is not None and len(df) else None
    return out


def _status_text(cfg: Config, cache: str) -> str:
    sr = SignalRunner(cfg, db_path=f"databases/{cfg.database}", cache_dir=cache)
    feed = DataFeed(cache, cfg.timeframe)
    op = sr.open_trades()
    cl = sr.closed_trades()
    realised = float(cl["pnl"].sum()) if len(cl) else 0.0

    prices = _live_prices(list(op["symbol"]), feed) if len(op) else {}
    unrealised = 0.0
    rows = []
    for _, r in op.head(30).iterrows():
        px = prices.get(r["symbol"])
        if px:
            unrealised += (px / r["open_price"] - 1) * r["stake"]
            pct_str = f"{(px / r['open_price'] - 1) * 100:+.1f}%"
        else:
            pct_str = "?"
        rows.append([r["symbol"], f"{r['open_price']:.2f}", f"{r['stake']:.0f}",
                     r["open_date"][:10], pct_str])
    tbl = (mono_table(["sym", "open", "stake", "date", "%"], rows)
          if rows else "no open positions")
    total = realised + unrealised
    return (f"<b>GIT_STOCK</b> · <code>{cfg.strategy}</code>  <i>(/strategy for params)</i>\n"
            f"{len(op)} open · {len(cl)} closed <i>(% is live, quotes ≤15min old)</i>\n{tbl}\n"
            f"realised £{realised:,.0f} · unrealised £{unrealised:+,.0f}\n"
            f"<b>cumulative profit: £{total:+,.0f}</b>")


def _price_ticker(cfg: Config, cache: str, interval: int = _QUOTE_TTL) -> None:
    """Background refresh of _quote_cache for currently-open symbols, every `interval`
    seconds, so /status never has to wait on a live fetch mid-request. Runs as a daemon
    thread INSIDE the bot process on purpose — same PID as the poll loop, so it starts
    and stops with the bot automatically (systemctl restart / kill both cleanly), instead
    of being a separate process that could get orphaned or fall out of sync."""
    feed = DataFeed(cache, cfg.timeframe)
    while True:
        try:
            sr = SignalRunner(cfg, db_path=f"databases/{cfg.database}", cache_dir=cache)
            symbols = list(sr.open_trades()["symbol"])
            if symbols:
                _live_prices(symbols, feed)
                log.info("price tick: refreshed %d symbol(s)", len(symbols))
        except Exception as e:  # noqa: BLE001
            log.warning("price ticker error: %s", e)
        time.sleep(interval)


def run_bot(cfg: Config, cache: str = "data_cache", poll: int = 25,
           config_path: str = "config/config.json") -> None:
    tg = cfg.telegram or {}
    token = tg.get("token", "")
    if not token:
        raise SystemExit("telegram token not set: put it in config/secrets.json (see secrets.example.json) "
                         "or GIT_STOCK_TELEGRAM_TOKEN")
    log.info("telegram bot polling…")
    if set_my_commands(token, _COMMANDS):
        log.info("registered %d commands in the Telegram menu", len(_COMMANDS))
    else:
        log.warning("could not register the Telegram command menu (non-fatal)")
    threading.Thread(target=_price_ticker, args=(cfg, cache), daemon=True).start()
    chat = tg.get("chat_id")
    if chat:
        name = tg.get("bot_name") or get_bot_username(token) or "GIT_STOCK bot"
        name = name if name.startswith("@") or " " in name else "@" + name
        send_message(token, chat, f"🤖 <b>{name}</b> online\nStrategy: <code>{_strategy_line(cfg)}</code>\n"
                                  "Commands: /status /strategy /symbols /report /chart SYMBOL [bars] "
                                  "/add SYMBOL SECTOR /delete SYMBOL [SECTOR]")
    offset = 0
    while True:
        try:
            updates = get_updates(token, offset=offset, timeout=poll)
        except Exception as e:  # noqa: BLE001
            log.warning("getUpdates: %s", e)
            time.sleep(5)
            continue
        for u in updates:
            offset = u["update_id"] + 1
            try:
                _handle(cfg, cache, token, u, config_path)
            except Exception as e:  # noqa: BLE001
                log.warning("handler error: %s", e)


def _handle(cfg: Config, cache: str, token: str, u: dict, config_path: str = "config/config.json") -> None:
    if "callback_query" in u:
        cq = u["callback_query"]
        data = cq.get("data", "")
        chat = cq["message"]["chat"]["id"]
        answer_callback(token, cq["id"], "rendering chart…")
        if data.startswith("chart:"):
            sym = data.split(":", 1)[1]
            png = _chart(cfg, cache, sym)
            if png:
                send_photo(token, chat, png, f"<b>{sym}</b> — {cfg.strategy}")
            else:
                send_message(token, chat, f"no data for {sym}")
        return

    msg = u.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat = (msg.get("chat") or {}).get("id")
    if not text or chat is None:
        return
    cmd, _, arg = text.partition(" ")
    cmd = cmd.lstrip("/").lower()

    if cmd in ("status", "start"):
        send_message(token, chat, _status_text(cfg, cache))
    elif cmd in ("strategy", "config"):
        s = get_strategy(cfg.strategy)({"window": cfg.timewindow, **(cfg.strategy_params or {})})
        rows = [[k, str(s.p.get(k))] for k in s.space]
        rows += [["timeframe", cfg.timeframe], ["stake", f"£{cfg.stake_amount:.0f}"],
                 ["max_open", str(cfg.max_open_trades)], ["universe", f"{len(cfg.tickers)} tickers"]]
        send_message(token, chat, f"<b>Strategy: {s.name}</b>\n" + mono_table(["param", "value"], rows))
    elif cmd == "report":
        sr = SignalRunner(cfg, db_path=f"databases/{cfg.database}", cache_dir=cache)
        cl = sr.closed_trades().tail(15)
        _why = {"stoploss": "SL", "trailing_stop": "trail", "exit_signal": "band", "take_profit": "TP"}
        rows = [[r["symbol"], r["close_date"][:10], f"{r['pnl']:+.2f}",
                 _why.get(r.get("exit_reason") or "", (r.get("exit_reason") or ""))] for _, r in cl.iterrows()]
        send_message(token, chat, mono_table(["sym", "closed", "P/L", "why"], rows) if rows else "no closed trades")
    elif cmd == "chart":
        bits = arg.split()
        if not bits:
            send_message(token, chat, "usage: /chart SYMBOL [bars]  e.g. /chart AAPL 250")
            return
        sym = bits[0].upper()
        lookback = 400
        if len(bits) > 1 and bits[1].isdigit():
            lookback = max(30, min(int(bits[1]), 4000))
        png = _chart(cfg, cache, sym, lookback)
        send_photo(token, chat, png, f"<b>{sym}</b> · {cfg.strategy} · last {lookback} bars") if png \
            else send_message(token, chat, f"no data for {sym}")
    elif cmd in ("symbols", "tickers", "universe"):
        feed = DataFeed(cache, cfg.timeframe)
        want = arg.strip().lower()
        if want and want in {s.lower() for s in cfg.sectors}:
            sec = next(s for s in cfg.sectors if s.lower() == want)
            names = cfg.sectors[sec]
            miss = [t for t in names if feed.load(t) is None]
            body = " ".join((f"{t}✗" if t in miss else t) for t in names)
            send_message(token, chat, f"<b>{sec}</b> ({len(names)})\n<pre>{body}</pre>")
        else:
            lines, missing = [], 0
            for sec, names in cfg.sectors.items():
                m = [t for t in names if feed.load(t) is None]
                missing += len(m)
                lines.append(f"{sec} ({len(names)}): " + " ".join(names))
            head = (f"<b>{len(cfg.tickers)} symbols</b> · {len(cfg.sectors)} sectors · "
                    f"{len(cfg.tickers) - missing} with data\n<i>/symbols SECTOR for one sector</i>")
            send_message(token, chat, head + "\n<pre>" + "\n".join(lines)[:3800] + "</pre>")
    elif cmd == "add":
        bits = arg.split()
        if len(bits) < 2:
            send_message(token, chat, "usage: /add SYMBOL SECTOR  e.g. /add SHOP retail")
            return
        sym, sector_in = bits[0].upper(), bits[1]
        existing = next((s for s in cfg.sectors if s.lower() == sector_in.lower()), None)
        sector = existing or sector_in.lower()
        already = cfg.sector_of(sym)
        if already:
            send_message(token, chat, f"{sym} is already in <b>{already}</b> — nothing to do "
                                      f"(use /delete {sym} first to move it)")
            return

        feed = DataFeed(cache, cfg.timeframe)
        if feed.load(sym) is None:
            feed.update([sym], cfg.daterange)
        has_data = feed.load(sym) is not None

        def _add(sectors: dict) -> None:
            sectors.setdefault(sector, [])
            if sym not in sectors[sector]:
                sectors[sector].append(sym)

        new_sectors = _edit_sectors(config_path, _add)
        cfg.sectors.clear()
        cfg.sectors.update(new_sectors)
        note = "" if has_data else " ⚠️ no price data found — check the ticker"
        created = "" if existing else " (new sector)"
        send_message(token, chat, f"✅ added <b>{sym}</b> to <b>{sector}</b>{created} — "
                                  f"universe is now {len(cfg.tickers)} symbols{note}")

    elif cmd == "delete":
        bits = arg.split()
        if not bits:
            send_message(token, chat, "usage: /delete SYMBOL [SECTOR]  e.g. /delete SHOP retail")
            return
        sym = bits[0].upper()
        sector_in = bits[1] if len(bits) > 1 else None

        sr = SignalRunner(cfg, db_path=f"databases/{cfg.database}", cache_dir=cache)
        if sym in set(sr.open_trades()["symbol"]):
            send_message(token, chat, f"⛔ {sym} has an OPEN position — refusing to delete it "
                                      "from the universe (it would stop being managed/exited). "
                                      "Wait for it to close, or handle it manually first.")
            return

        target_sector = None
        if sector_in:
            target_sector = next((s for s in cfg.sectors if s.lower() == sector_in.lower()), sector_in)

        removed_from = []

        def _delete(sectors: dict) -> None:
            for sec, names in sectors.items():
                if target_sector and sec != target_sector:
                    continue
                if sym in names:
                    names.remove(sym)
                    removed_from.append(sec)

        new_sectors = _edit_sectors(config_path, _delete)
        cfg.sectors.clear()
        cfg.sectors.update(new_sectors)
        if not removed_from:
            where = f" in <b>{target_sector}</b>" if target_sector else ""
            send_message(token, chat, f"{sym} not found{where} — nothing removed")
        else:
            send_message(token, chat, f"🗑 removed <b>{sym}</b> from {', '.join(removed_from)} — "
                                      f"universe is now {len(cfg.tickers)} symbols")

    elif cmd == "help":
        send_message(token, chat, f"Running <code>{_strategy_line(cfg)}</code>\n\n"
                                  "/status — portfolio + strategy\n/strategy — full config\n"
                                  "/report — closed trades\n/symbols [SECTOR] — the universe\n"
                                  "/chart SYMBOL [bars] — signal chart (bars optional, default 400)\n"
                                  "/add SYMBOL SECTOR — add a ticker (new or existing sector)\n"
                                  "/delete SYMBOL [SECTOR] — remove a ticker (all sectors if omitted; "
                                  "blocked while it has an open position)")
