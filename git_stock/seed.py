"""One-shot import of a starting paper portfolio from the Finance dashboard's
``portfolio.yaml``.

This is a **copy, not a link** — it reads the file once to give the signal bot a
realistic starting book (stocks + funds/ETFs only). Nothing is written back and the
dashboard is never contacted again.
"""
from __future__ import annotations

import datetime as dt
import logging

log = logging.getLogger("git_stock.seed")

EQUITY_CLASSES = {"stock", "etf", "fund"}


def _fx(frm: str, to: str) -> float:
    frm, to = frm.upper(), to.upper()
    stable = {"USDT", "USDC", "USD"}
    if frm == to or (frm in stable and to in stable):
        return 1.0
    a = "USD" if frm in stable else frm
    b = "USD" if to in stable else to
    if a == b:
        return 1.0
    try:
        import yfinance as yf

        h = yf.Ticker(f"{a}{b}=X").history(period="5d")
        if not h.empty:
            return float(h["Close"].dropna().iloc[-1])
        h = yf.Ticker(f"{b}{a}=X").history(period="5d")
        if not h.empty:
            return 1.0 / float(h["Close"].dropna().iloc[-1])
    except Exception as e:  # noqa: BLE001
        log.warning("fx %s%s failed: %s", a, b, e)
    return 1.0


def _quote(symbol: str) -> float | None:
    try:
        import yfinance as yf

        h = yf.Ticker(symbol).history(period="7d")
        if h.empty:
            return None
        px = float(h["Close"].dropna().iloc[-1])
        # LSE quotes come back in pence
        cur = (yf.Ticker(symbol).fast_info.get("currency") or "").strip()
        return px / 100.0 if cur in ("GBp", "GBX") else px
    except Exception:  # noqa: BLE001
        return None


def positions_from_yaml(path: str, base: str = "GBP") -> list[dict]:
    import yaml

    raw = yaml.safe_load(open(path)) or {}
    out = []
    for p in raw.get("positions", []):
        if p.get("asset_class") not in EQUITY_CLASSES:
            continue
        sym = p.get("yahoo") or p.get("quote_symbol") or p.get("symbol")
        if not sym:
            continue
        # value in base currency
        if p.get("net_value") is not None:
            ccy = (p.get("net_value_currency") or base).upper()
            value = float(p["net_value"]) * _fx(ccy, base)
        elif p.get("quantity") is not None and p.get("avg_price") is not None:
            ccy = (p.get("cost_currency") or base).upper()
            value = float(p["quantity"]) * float(p["avg_price"]) * _fx(ccy, base)
        elif p.get("cost_basis") is not None:
            ccy = (p.get("cost_currency") or base).upper()
            value = float(p["cost_basis"]) * _fx(ccy, base)
        else:
            continue
        px = _quote(sym)
        priceable = px is not None
        if px is None and p.get("quantity"):
            px = value / float(p["quantity"])
        out.append({
            "symbol": sym, "account": p.get("account", ""),
            "asset_class": p.get("asset_class"), "name": p.get("name", ""),
            "value_base": round(value, 2), "price": round(px, 4) if px else None,
            "priceable": priceable,
        })
    return out


def seed_ledger(runner, positions: list[dict], force: bool = False) -> list[dict]:
    """Insert the positions as `open` paper trades. Refuses if the ledger already
    holds open trades unless force=True."""
    conn = runner._conn()
    try:
        existing = conn.execute("SELECT COUNT(*) FROM trades WHERE status='open'").fetchone()[0]
        if existing and not force:
            raise RuntimeError(f"ledger already has {existing} open trades — use force to reseed")
        today = dt.date.today().isoformat()
        done = []
        for p in positions:
            price = p["price"] or p["value_base"]
            conn.execute(
                "INSERT INTO trades(symbol,sector,strategy,open_date,open_price,stake,"
                "open_fee,status) VALUES (?,?,?,?,?,?,0,'open')",
                (p["symbol"], p["account"], "seed", today, price, p["value_base"]))
            done.append(p)
        conn.commit()
        return done
    finally:
        conn.close()
