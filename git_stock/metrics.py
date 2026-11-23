"""Performance / risk statistics from an equity curve + a trades table."""
from __future__ import annotations

import numpy as np
import pandas as pd

_PERIODS = {"1d": 252, "1h": 252 * 7, "15m": 252 * 7 * 4, "5m": 252 * 7 * 12, "1m": 252 * 7 * 60}


def _annualise(equity: pd.Series, timeframe: str) -> tuple[float, float, float]:
    rets = equity.pct_change().dropna()
    if len(rets) < 2:
        return 0.0, 0.0, 0.0
    ppy = _PERIODS.get(timeframe, 252)
    years = len(rets) / ppy
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 and equity.iloc[0] > 0 else 0.0
    vol = rets.std(ddof=1) * np.sqrt(ppy)
    down = rets[rets < 0]
    dvol = down.std(ddof=1) * np.sqrt(ppy) if len(down) > 1 else 0.0
    return cagr, vol, dvol


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def max_drawdown_duration(equity: pd.Series) -> tuple[int, float]:
    """Longest peak-to-full-recovery span around the worst drawdown, as (bar count,
    fraction of the series length). If that drawdown never recovers within the series,
    duration runs to the series end (a conservative, not optimistic, measure) - this is
    what lets a hyperopt objective penalise a long, sustained losing stretch, which
    Sharpe/Sortino alone cannot see: those ratios depend only on the mean/dispersion of
    returns, not on when within the window they occurred or how long any one drawdown
    lasted (see the CleciDecomposed case study)."""
    if len(equity) < 2:
        return 0, 0.0
    peak = equity.cummax()
    dd = (equity / peak - 1.0).to_numpy()
    trough_pos = int(np.argmin(dd))
    peak_level = peak.iloc[trough_pos]
    pre = equity.to_numpy()[:trough_pos + 1]
    peak_pos = len(pre) - 1 - int(np.argmax(pre[::-1] >= peak_level))
    post = equity.to_numpy()[trough_pos:]
    recovered = post >= peak_level
    recovery_pos = trough_pos + int(np.argmax(recovered)) if recovered.any() else len(equity) - 1
    duration_bars = recovery_pos - peak_pos
    return duration_bars, duration_bars / len(equity)


def stats(equity: pd.Series, trades: pd.DataFrame, initial: float,
          timeframe: str = "1d", rf: float = 0.02) -> dict:
    end = float(equity.iloc[-1]) if len(equity) else initial
    total_return = end / initial - 1.0 if initial else 0.0
    cagr, vol, dvol = _annualise(equity, timeframe)
    mdd = max_drawdown(equity)
    dd_duration_bars, dd_duration_frac = max_drawdown_duration(equity)

    wins = trades[trades["pnl"] > 0] if len(trades) else trades
    losses = trades[trades["pnl"] <= 0] if len(trades) else trades
    gross_win = float(wins["pnl"].sum()) if len(wins) else 0.0
    gross_loss = float(-losses["pnl"].sum()) if len(losses) else 0.0

    # The backtester force-closes every still-open position at the window's last known
    # price (exit_reason "end") so total_return/equity/Sharpe/maxDD always reflect the
    # true mark-to-market result — nothing is left dangling off the equity curve. But
    # those "end" exits were never a real, strategy-chosen exit: they're whatever the
    # position happened to be worth on the arbitrary last bar of the test window, which
    # can flatter or understate a strategy if a big winner/loser just hadn't hit its own
    # exit yet. Split the P&L so that's visible rather than hidden inside one return %.
    if len(trades):
        forced = trades["exit_reason"] == "end"
        realised_pnl = float(trades.loc[~forced, "pnl"].sum())
        unrealised_pnl = float(trades.loc[forced, "pnl"].sum())
        open_at_end = int(forced.sum())
    else:
        realised_pnl = unrealised_pnl = 0.0
        open_at_end = 0
    total_pnl = realised_pnl + unrealised_pnl
    unrealised_frac = round(abs(unrealised_pnl) / abs(total_pnl), 3) if total_pnl else 0.0

    return {
        "start_balance": round(initial, 2),
        "end_balance": round(end, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "sharpe": round((cagr - rf) / vol, 2) if vol else None,
        "sortino": round((cagr - rf) / dvol, 2) if dvol else None,
        "calmar": round(cagr / abs(mdd), 2) if mdd else None,
        "max_drawdown_pct": round(mdd * 100, 2),
        "dd_duration_bars": dd_duration_bars,
        "dd_duration_frac": round(dd_duration_frac, 3),   # of this series' own length
        "trades": int(len(trades)),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if len(trades) else 0.0,
        "avg_trade_pct": round(float(trades["pnl_pct"].mean()) * 100, 2) if len(trades) else 0.0,
        "avg_win_pct": round(float(wins["pnl_pct"].mean()) * 100, 2) if len(wins) else 0.0,
        "avg_loss_pct": round(float(losses["pnl_pct"].mean()) * 100, 2) if len(losses) else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "expectancy": round(float(trades["pnl"].mean()), 2) if len(trades) else 0.0,
        "avg_bars_held": round(float(trades["bars_held"].mean()), 1) if len(trades) else 0.0,
        "best_trade_pct": round(float(trades["pnl_pct"].max()) * 100, 2) if len(trades) else 0.0,
        "worst_trade_pct": round(float(trades["pnl_pct"].min()) * 100, 2) if len(trades) else 0.0,
        # realised (genuine stoploss/trailing_stop/take_profit/exit_signal exits) vs
        # unrealised-at-cutoff ("end": still open when the test window ran out).
        # realised_pct + unrealised_pct == total_return_pct, always.
        "realised_pct": round(realised_pnl / initial * 100, 2) if initial else 0.0,
        "unrealised_pct": round(unrealised_pnl / initial * 100, 2) if initial else 0.0,
        "open_at_end": open_at_end,
        "unrealised_frac": unrealised_frac,   # |unrealised P&L| / |total P&L|, 0..1
    }
