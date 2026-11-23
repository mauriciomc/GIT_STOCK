"""Render a price chart with the current strategy's bands + entry/exit markers to PNG."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_signals(prepared: pd.DataFrame, symbol: str, strategy_name: str,
                 lookback: int = 400, real_trades: list[dict] | None = None) -> bytes:
    """`prepared` is a strategy.prepare(df) frame (has close, entry, exit, and any
    indicator columns the strategy added).

    Marker design: raw entry/exit *signal* crossings (every day the condition was true)
    are shown as small, faint markers purely for context — "here's where the algorithm
    could plausibly have acted." They are NOT resolved into a simulated single-position
    walk (that used to be done via a from-day-one `actionable_signals()` replay, which
    silently drifts from reality: the live/backtest engines' stop-loss and trailing-stop
    exits aren't modelled in the raw exit-signal column, so an old, never-really-closed
    phantom position could swallow a real, recent entry and hide it from the chart).

    Instead, what's ACTUALLY open or closed is drawn from `real_trades` — the ledger's
    own recorded open_date/close_date for this symbol — so the chart always agrees with
    `/status` and `/report`, however long ago the signal that triggered it happened.
    `real_trades`: list of dicts with open_date, open_price, close_date (None if still
    open), close_price (None if still open), pnl_pct (None if still open).
    """
    df = prepared.tail(lookback)
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"hspace": 0.05, "height_ratios": [4, 1]})

    ax.plot(df.index, df["close"], lw=1.3, color="#1f77b4", label="close")
    # entry_threshold/exit_threshold (band * entry_mult/exit_mult) are the REAL
    # levels entry_signal/exit_signal compare `close` against - plotted in
    # preference to the raw band_low/band_high, which sit 16-30% away from the
    # actual trigger and make the chart look inconsistent with its own signal
    # markers. Only strategies with a multiplier (Cleci and its subclasses)
    # have these; fall back to the raw bands for anything else.
    band_cols = (
        [("entry_threshold", "#2ca02c"), ("exit_threshold", "#d62728")]
        if "entry_threshold" in df.columns else
        [("band_low", "#2ca02c"), ("band_high", "#d62728")]
    )
    for band, style in (*band_cols, ("bb_low", "#2ca02c"), ("bb_high", "#d62728"),
                        ("ema_fast", "#ff7f0e"), ("ema_slow", "#9467bd")):
        if band in df.columns:
            ax.plot(df.index, df[band], lw=0.9, ls="--", color=style, alpha=0.8, label=band)

    # raw signal crossings: every day the condition fired, shown small + faint, no
    # position-simulation applied — purely "the algorithm looked interested here"
    raw_e = df[df["entry"].to_numpy(bool)]
    raw_x = df[df["exit"].to_numpy(bool)]
    ax.scatter(raw_e.index, raw_e["close"], marker="^", s=28, color="#2ca02c",
               alpha=0.45, zorder=3, label=f"entry signal ({len(raw_e)})")
    ax.scatter(raw_x.index, raw_x["close"], marker="v", s=28, color="#d62728",
               alpha=0.45, zorder=3, label=f"exit signal ({len(raw_x)})")

    # real trades: what the ledger actually recorded, drawn bold and dated exactly —
    # this is the source of truth, independent of how the signal replay reads history
    real_trades = [t for t in (real_trades or [])
                   if pd.Timestamp(t["open_date"]) >= df.index[0]
                   or (t.get("close_date") and pd.Timestamp(t["close_date"]) >= df.index[0])]
    for t in real_trades:
        open_dt = pd.Timestamp(t["open_date"])
        close_dt = pd.Timestamp(t["close_date"]) if t.get("close_date") else df.index[-1]
        is_open = t.get("close_date") is None
        span_color = "#2ca02c" if (is_open or (t.get("pnl_pct") or 0) >= 0) else "#d62728"
        ax.axvspan(max(open_dt, df.index[0]), close_dt, color=span_color, alpha=0.08, zorder=1)
        if open_dt >= df.index[0]:
            ax.scatter([open_dt], [t["open_price"]], marker="^", s=140, color="#1a7f1a",
                       edgecolors="white", linewidths=1, zorder=6,
                       label="_nolegend_")
            ax.annotate(f"open {open_dt.date()}", (open_dt, t["open_price"]),
                        textcoords="offset points", xytext=(0, 12), fontsize=7.5,
                        ha="center", color="#1a7f1a", fontweight="bold")
        if not is_open:
            pnl_txt = f"{t['pnl_pct']:+.1f}%" if t.get("pnl_pct") is not None else ""
            ax.scatter([close_dt], [t["close_price"]], marker="v", s=140, color="#a11a1a",
                       edgecolors="white", linewidths=1, zorder=6, label="_nolegend_")
            ax.annotate(f"close {close_dt.date()} {pnl_txt}", (close_dt, t["close_price"]),
                        textcoords="offset points", xytext=(0, -14), fontsize=7.5,
                        ha="center", color="#a11a1a", fontweight="bold")

    ax.set_title(f"{symbol} — {strategy_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.grid(alpha=0.25)

    if "volume" in df.columns:
        axv.bar(df.index, df["volume"], width=1.0, color="#888", alpha=0.6)
    axv.grid(alpha=0.25)
    axv.set_ylabel("vol", fontsize=8)
    axv.xaxis.set_major_locator(mdates.AutoDateLocator())
    axv.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axv.xaxis.get_major_locator()))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
