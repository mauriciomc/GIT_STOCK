"""Live (paper) signal generation — the daily runner.

Reads the latest candle for every configured ticker, and for the chosen strategy
opens / closes **paper** positions in a SQLite ledger.  No broker connection — it
emits signals; you (or, later, an execution layer) act on them.
"""
from __future__ import annotations

import logging
import pathlib
import sqlite3
from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestConfig
from .config import Config
from .data import DataFeed
from .strategies import get as get_strategy

log = logging.getLogger("git_stock.signals")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY,
    symbol       TEXT NOT NULL,
    sector       TEXT,
    strategy     TEXT NOT NULL,
    open_date    TEXT NOT NULL,
    close_date   TEXT,
    open_price   REAL NOT NULL,
    close_price  REAL,
    stake        REAL NOT NULL,
    open_fee     REAL NOT NULL,
    close_fee    REAL,
    pnl          REAL,
    status       TEXT NOT NULL DEFAULT 'open',
    exit_reason  TEXT
);
"""


@dataclass
class RunResult:
    opened: list[dict]
    closed: list[dict]
    open_count: int
    balance: float
    invested: float
    equity: float
    realised_pnl: float
    strategy: str = ""

    def as_text(self) -> str:
        from .notify import mono_table

        head = (f"<b>GIT_STOCK</b> · <code>{self.strategy}</code>\n"
                f"equity <b>£{self.equity:,.0f}</b>  ·  cash £{self.balance:,.0f}  ·  "
                f"{self.open_count} open  ·  realised £{self.realised_pnl:,.0f}")
        blocks = [head]
        if self.closed:
            why = {"stoploss": "SL", "trailing_stop": "trail", "exit_signal": "band",
                   "take_profit": "TP"}
            rows = [[t["symbol"], f"{t['close_price']:.2f}", f"{t['pnl']:+.2f}",
                     f"{t['pnl_pct']:+.1f}%", why.get(t.get("exit_reason", ""), t.get("exit_reason", ""))]
                    for t in self.closed]
            blocks.append("🔴 <b>Closed</b>\n" + mono_table(["sym", "price", "P/L", "%", "why"], rows))
        if self.opened:
            rows = [[t["symbol"], f"{t['open_price']:.2f}", f"£{t['stake']:.0f}"]
                    for t in self.opened]
            blocks.append("🟢 <b>Opened</b>\n" + mono_table(["sym", "price", "stake"], rows))
        if not self.opened and not self.closed:
            blocks.append("no new signals")
        return "\n\n".join(blocks)

    def buttons(self) -> list[list[dict]]:
        syms, seen = [], set()
        for t in self.closed + self.opened:
            s = t["symbol"]
            if s not in seen:
                seen.add(s)
                syms.append(s)
        return [[{"text": f"📈 {s}", "callback_data": f"chart:{s}"} for s in syms[i:i + 3]]
                for i in range(0, len(syms[:12]), 3)]


class SignalRunner:
    def __init__(self, cfg: Config, db_path: str = "databases/signals.db",
                 cache_dir: str = "data_cache"):
        self.cfg = cfg
        self.db_path = pathlib.Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.feed = DataFeed(cache_dir, cfg.timeframe)
        self.strategy = get_strategy(cfg.strategy)(
            {"window": cfg.timewindow, **(cfg.strategy_params or {})})
        self.bt = BacktestConfig(
            initial_balance=cfg.initial_balance, stake_amount=cfg.stake_amount,
            max_open_trades=cfg.max_open_trades, fee=cfg.open_fee,
            stoploss_pct=cfg.stop_loss, take_profit_pct=cfg.take_profit,
            timeframe=cfg.timeframe)

    # ---- ledger ------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        c.executescript(_SCHEMA)
        # migrate pre-exit_reason ledgers in place (CREATE TABLE IF NOT EXISTS won't)
        cols = {r[1] for r in c.execute("PRAGMA table_info(trades)")}
        if "exit_reason" not in cols:
            c.execute("ALTER TABLE trades ADD COLUMN exit_reason TEXT")
            c.commit()
        return c

    def _open_positions(self, conn) -> dict[str, sqlite3.Row]:
        return {r["symbol"]: r for r in conn.execute(
            "SELECT * FROM trades WHERE status='open'")}

    def _realised(self, conn) -> float:
        v = conn.execute("SELECT COALESCE(SUM(pnl),0) FROM trades WHERE status='closed'").fetchone()[0]
        return float(v or 0.0)

    def _last_exit_date(self, conn, symbol: str) -> pd.Timestamp | None:
        v = conn.execute(
            "SELECT MAX(close_date) FROM trades WHERE symbol=? AND status='closed'", (symbol,)
        ).fetchone()[0]
        return pd.Timestamp(v) if v else None

    def _cooldown_clear(self, conn, symbol: str, last_bar_date: pd.Timestamp) -> bool:
        """Mirrors backtest.py's post-exit cooldown: a symbol must sit out `cooldown_bars`
        bars after its last close (any reason) before it's allowed to re-enter."""
        cooldown = getattr(self.strategy, "cooldown_bars", None) or 0
        if cooldown <= 0:
            return True
        last_exit = self._last_exit_date(conn, symbol)
        if last_exit is None:
            return True
        bars_since = (last_bar_date - last_exit).days   # daily-bar strategies only, same as elsewhere
        return bars_since >= cooldown

    # ---- run -------------------------------------------------------
    def refresh_data(self) -> None:
        self.feed.update(self.cfg.tickers, self.cfg.daterange)

    def _stop_or_trail_exit(self, prep: pd.DataFrame, open_price: float,
                            open_date: str) -> tuple[float, "pd.Timestamp", str] | None:
        """Check whether a stop-loss or trailing-stop would have fired on any bar since
        the position opened — mirrors backtest.py's per-bar checks exactly (same order:
        stop-loss first, then trailing, same high-water-mark tracked from the entry
        bar's own high, same "trail only counts once it's above the hard stop" guard),
        so live behaviour actually matches what was backtested. Without this, only the
        band-crossing `exit_signal()` column was ever checked here, and any strategy
        that relies on its stop/trail for most exits (e.g. clecitrail) would just
        accumulate open positions forever. Returns the FIRST bar it triggers on, since
        a position may be catching up on days it wasn't checked (e.g. right after this
        fix ships, or after a gap in the daily job)."""
        strat = self.strategy
        sl = (strat.stoploss_pct if strat.stoploss_pct is not None else self.bt.stoploss_pct) / 100.0
        trail_pct = (getattr(strat, "trailing_stop_pct", None)
                    if getattr(strat, "trailing_stop_pct", None) is not None
                    else self.bt.trailing_stop_pct) / 100.0
        trail_activation = (getattr(strat, "trailing_activation_pct", None)
                            if getattr(strat, "trailing_activation_pct", None) is not None
                            else getattr(self.bt, "trailing_activation_pct", 0.0)) / 100.0
        since = prep.loc[prep.index >= pd.Timestamp(open_date)]
        if len(since) < 2:
            return None
        sl_price = open_price * (1 - sl)
        hw = float(since["high"].iloc[0])          # seeded from the entry bar's high
        for dt, row in since.iloc[1:].iterrows():  # never checked on the entry bar itself
            hw = max(hw, float(row["high"]))
            if float(row["low"]) <= sl_price:
                return sl_price, dt, "stoploss"
            if trail_pct and hw >= open_price * (1 + trail_activation):
                trail_price = hw * (1 - trail_pct)
                if float(row["low"]) <= trail_price and trail_price > sl_price:
                    return trail_price, dt, "trailing_stop"
        return None

    def run(self, update: bool = True) -> RunResult:
        if update:
            self.refresh_data()
        data = self.feed.load_many(self.cfg.tickers)
        conn = self._conn()
        try:
            open_pos = self._open_positions(conn)
            invested = sum(r["stake"] for r in open_pos.values())
            balance = self.cfg.initial_balance + self._realised(conn) - invested
            opened, closed = [], []

            for sym, df in data.items():
                prep = self.strategy.prepare(df)
                if prep.empty:
                    continue
                last = prep.iloc[-1]
                price = float(last["close"])

                exit_price = exit_date = exit_reason = None
                if sym in open_pos:
                    r = open_pos[sym]
                    stop = self._stop_or_trail_exit(prep, r["open_price"], r["open_date"])
                    if stop is not None:
                        exit_price, exit_date, exit_reason = stop
                    elif bool(last["exit"]):
                        exit_price, exit_date, exit_reason = price, prep.index[-1], "exit_signal"

                if exit_price is not None:
                    r = open_pos[sym]
                    qty = r["stake"] / r["open_price"]
                    gross = qty * exit_price
                    fee = gross * self.cfg.close_fee
                    pnl = (gross - fee) - r["stake"] * (1 + r["open_fee"])
                    conn.execute(
                        "UPDATE trades SET status='closed', close_date=?, close_price=?, "
                        "close_fee=?, pnl=?, exit_reason=? WHERE id=?",
                        (str(exit_date), exit_price, fee, pnl, exit_reason, r["id"]))
                    balance += gross - fee
                    closed.append({"symbol": sym, "close_price": exit_price, "pnl": pnl,
                                   "pnl_pct": pnl / r["stake"] * 100, "exit_reason": exit_reason})
                elif sym not in open_pos and bool(last["entry"]) \
                        and len(open_pos) + len(opened) < self.cfg.max_open_trades \
                        and balance >= self.cfg.stake_amount \
                        and self._cooldown_clear(conn, sym, prep.index[-1]):
                    conn.execute(
                        "INSERT INTO trades(symbol,sector,strategy,open_date,open_price,"
                        "stake,open_fee,status) VALUES (?,?,?,?,?,?,?, 'open')",
                        (sym, self.cfg.sector_of(sym), self.strategy.name,
                         str(prep.index[-1]), price, self.cfg.stake_amount, self.cfg.open_fee))
                    balance -= self.cfg.stake_amount * (1 + self.cfg.open_fee)
                    opened.append({"symbol": sym, "open_price": price,
                                   "stake": self.cfg.stake_amount})
            conn.commit()

            open_pos = self._open_positions(conn)
            invested = sum(r["stake"] for r in open_pos.values())
            mtm = 0.0
            for sym, r in open_pos.items():
                d = data.get(sym)
                px = float(d["close"].iloc[-1]) if d is not None else r["open_price"]
                mtm += (r["stake"] / r["open_price"]) * px
            realised = self._realised(conn)
            balance = self.cfg.initial_balance + realised - invested
            return RunResult(opened, closed, len(open_pos), balance, invested,
                             balance + mtm, realised, strategy=self.strategy.describe())
        finally:
            conn.close()

    def open_trades(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query("SELECT * FROM trades WHERE status='open' ORDER BY open_date", conn)

    def closed_trades(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query("SELECT * FROM trades WHERE status='closed' ORDER BY close_date", conn)

    def trades_for(self, symbol: str) -> list[dict]:
        """What the ledger actually recorded for one symbol — open + closed — in the
        shape charts.plot_signals() wants. This is the source of truth for "is this
        position open/closed and since when", independent of any signal-column replay."""
        trades = []
        op = self.open_trades()
        for _, r in op[op["symbol"] == symbol].iterrows():
            trades.append({"open_date": r["open_date"], "open_price": r["open_price"],
                           "close_date": None, "close_price": None, "pnl_pct": None})
        cl = self.closed_trades()
        for _, r in cl[cl["symbol"] == symbol].iterrows():
            pnl_pct = (r["pnl"] / r["stake"] * 100) if r.get("stake") else None
            trades.append({"open_date": r["open_date"], "open_price": r["open_price"],
                           "close_date": r["close_date"], "close_price": r["close_price"],
                           "pnl_pct": pnl_pct})
        return trades
