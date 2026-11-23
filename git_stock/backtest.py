"""Vectorised-prep, event-loop backtester.

Assumptions (documented so results are reproducible):
* long-only; one open position per symbol at a time
* by default entry and signal-exit fill at the signal candle's **close**; with
  ``fill_next_open`` they fill at the *next* bar's open (no same-bar look-ahead)
* stop-loss / trailing-stop fill at the stop level if the candle's low pierces it
  (checked first); with ``gap_stops`` a bar that *opens* through the level fills at
  the open instead (overnight-gap risk)
* take-profit fills at the target price if the candle's high reaches it (resting limit)
* costs per fill:  ``fee`` (commission, per side) + ``slippage_pct`` + ``spread_pct``/2
  applied adversely to the price (buys fill higher, sells fill lower)
* positions still open at the end are closed at the last close (also cost-charged)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics
from .strategy import Strategy


@dataclass
class BacktestConfig:
    initial_balance: float = 1000.0
    stake_amount: float = 40.0
    max_open_trades: int = 40
    fee: float = 0.001                    # commission per side (fraction of notional)
    stoploss_pct: float = 5.0
    take_profit_pct: float = 100.0
    trailing_stop_pct: float = 0.0        # 0 = off; e.g. 5.0 = exit 5% below the high-water mark
    trailing_activation_pct: float = 0.0  # trail only arms once unrealised gain reaches this
                                           # much above entry; 0 = armed immediately (legacy)
    cooldown_bars: int = 0                # bars a symbol must sit out after ANY exit (signal,
                                           # stoploss, trailing, take-profit) before re-entering;
                                           # 0 = off (legacy)
    timeframe: str = "1d"
    # ---- friction knobs (all default OFF so legacy results are unchanged) ----
    slippage_pct: float = 0.0            # adverse price move per fill (market-order slip)
    spread_pct: float = 0.0             # full bid/ask spread; half is paid per side
    fill_next_open: bool = False        # signals act on the NEXT bar's open, not this close
    gap_stops: bool = False             # stop/trail fill at the open when price gapped through
    # ---- portfolio rotation (default OFF) ----
    # when at max_open_trades and a new entry signal fires, force-close the weakest-scoring
    # eligible open position (strategy.score(), see strategy.py) to free a slot for the new
    # candidate, IF the candidate clears it by rotate_margin_pct. Only engages for strategies
    # that provide a real score (default score() is all-NaN => rotation never triggers).
    rotate: bool = False
    rotate_margin_pct: float = 1.0      # required score edge (percentage points) to force a swap
    min_hold_bars: int = 5              # a position can't be rotated out before this many bars
    # ---- market-regime filter (default OFF) ----
    # pause NEW entries (existing positions still managed normally) while a precomputed
    # market-wide drawdown series is below -regime_dd_threshold%. regime_dd is set ONCE
    # by the caller (e.g. an equal-weighted composite of the whole universe) and carried
    # through unchanged across hyperopt trials via dataclasses.replace.
    regime_dd_threshold: float = 0.0    # 0 = off (legacy); e.g. 15.0 = no new entries while
                                         # the market proxy is more than 15% off its high
    regime_dd: object = None            # pd.Series of drawdown (negative fractions), date-indexed
    # ---- per-sector stake multiplier (default OFF) ----
    # effective_stake = stake_amount * sector_weights[sector_of[symbol]] (missing sector or
    # symbol => weight 1.0, unchanged). Both dicts are set ONCE by the caller and carried
    # through unchanged across hyperopt trials via dataclasses.replace, same as regime_dd.
    sector_weights: object = None       # dict[str, float], sector name -> multiplier
    sector_of: object = None            # dict[str, str], symbol -> sector name

    @property
    def _slip(self) -> float:
        """Effective one-sided adverse price fraction (slippage + half the spread)."""
        return self.slippage_pct + self.spread_pct / 2.0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    stats: dict
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = self.stats
        return (f"{s['trades']} trades · return {s['total_return_pct']}% · "
                f"CAGR {s['cagr_pct']}% · Sharpe {s['sharpe']} · "
                f"maxDD {s['max_drawdown_pct']}% · win {s['win_rate_pct']}%")


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def _risk_of(strategy: Strategy, cfg: BacktestConfig) -> dict:
    """Extract the risk-management knobs (sl/tp/trail/...) that a single Strategy
    instance carries, falling back to BacktestConfig's generic defaults for any it
    doesn't override. Factored out so per-sector mode (run() given a dict of
    sector -> Strategy) can resolve these once per sector instead of once globally."""
    return {
        "sl": (strategy.stoploss_pct if strategy.stoploss_pct is not None else cfg.stoploss_pct) / 100.0,
        "tp": (strategy.take_profit_pct if strategy.take_profit_pct is not None else cfg.take_profit_pct) / 100.0,
        "trail": (getattr(strategy, "trailing_stop_pct", None)
                 if getattr(strategy, "trailing_stop_pct", None) is not None
                 else cfg.trailing_stop_pct) / 100.0,
        "trail_activation": (getattr(strategy, "trailing_activation_pct", None)
                            if getattr(strategy, "trailing_activation_pct", None) is not None
                            else cfg.trailing_activation_pct) / 100.0,
        "trail_activation_from": getattr(strategy, "trail_activation_from", None) or "entry",
        "cooldown": (getattr(strategy, "cooldown_bars", None)
                    if getattr(strategy, "cooldown_bars", None) is not None
                    else cfg.cooldown_bars),
    }


def run(data: dict[str, pd.DataFrame], strategy, cfg: BacktestConfig,
        start=None, end=None) -> BacktestResult:
    """``strategy`` is normally a single Strategy instance, applied to every symbol
    (unchanged behaviour). Pass a ``dict[sector_name, Strategy]`` instead to give
    each sector its own fully independent strategy instance - different indicator
    params AND different sl/tp/trail/cooldown per sector, not just a stake-size
    weight (that's still sector_weights on ``cfg``, and composes fine with this).
    A symbol whose sector isn't a key in the dict falls back to the dict's first
    value. cfg.sector_of maps symbol -> sector name for this lookup."""
    per_sector = isinstance(strategy, dict)
    sector_of_early = cfg.sector_of or {}
    default_strategy = next(iter(strategy.values())) if per_sector else strategy

    def _strategy_for(sym: str) -> Strategy:
        if not per_sector:
            return strategy
        return strategy.get(sector_of_early.get(sym, ""), default_strategy)

    regime_threshold = cfg.regime_dd_threshold / 100.0
    regime_dd = cfg.regime_dd if regime_threshold > 0 else None
    sector_weights = cfg.sector_weights or {}
    sector_of = cfg.sector_of or {}

    def _stake_for(sym: str, dtv=None) -> float:
        """sector_weights[sector] is either a flat float (same weight all year) or a
        dict {1: w_q1, 2: w_q2, 3: w_q3, 4: w_q4} for seasonal, per-quarter weights -
        looked up from dtv's calendar month. Falls back to 1.0 (neutral) if unknown."""
        w = sector_weights.get(sector_of.get(sym, ""), 1.0)
        if isinstance(w, dict):
            q = (pd.Timestamp(dtv).month - 1) // 3 + 1 if dtv is not None else 1
            w = w.get(q, w.get(str(q), 1.0))
        return cfg.stake_amount * w

    fee = cfg.fee
    slip = cfg._slip
    next_open = cfg.fill_next_open
    gap = cfg.gap_stops
    rotate = cfg.rotate
    rotate_margin = cfg.rotate_margin_pct / 100.0

    prepped: dict[str, dict] = {}
    risk_by_sym: dict[str, dict] = {}
    for sym, df in data.items():
        sym_strategy = _strategy_for(sym)
        risk_by_sym[sym] = _risk_of(sym_strategy, cfg)
        d = _slice(sym_strategy.prepare(df), start, end)
        if len(d) < 5:
            continue
        ent, ex = d["entry"], d["exit"]
        if next_open:                      # seen at close of bar i-1, acted on open of bar i
            ent = ent.shift(1, fill_value=False)
            ex = ex.shift(1, fill_value=False)
        prepped[sym] = {
            "open": d["open"].to_numpy(float),
            "close": d["close"].to_numpy(float),
            "high": d["high"].to_numpy(float),
            "low": d["low"].to_numpy(float),
            "entry": ent.to_numpy(bool),
            "exit": ex.to_numpy(bool),
            "score": d["score"].to_numpy(float) if "score" in d.columns else None,
            "exit_threshold": d["exit_threshold"].to_numpy(float) if "exit_threshold" in d.columns else None,
            "dates": d.index.to_numpy(),
        }
    if not prepped:
        raise ValueError("no usable data in the requested range")

    events: list[tuple] = []
    for sym, a in prepped.items():
        for i, dtv in enumerate(a["dates"]):
            events.append((dtv, sym, i))
    events.sort(key=lambda e: e[0])

    cash = cfg.initial_balance
    open_pos: dict[str, dict] = {}
    last_close: dict[str, float] = {}
    last_score: dict[str, float] = {}
    last_exit_day: dict[str, int] = {}
    closed: list[dict] = []
    eq_dates: list = []
    eq_vals: list[float] = []
    cur_day = None
    day_idx = -1        # universal day counter (unlike each symbol's own bar index `i`,
                         # comparable across symbols for the rotation cooldown below)

    def mark_equity(dtv):
        held = sum(p["qty"] * last_close.get(s, p["entry_price"]) for s, p in open_pos.items())
        eq_dates.append(pd.Timestamp(dtv))
        eq_vals.append(cash + held)

    def _sell(p, raw_px, reason, dtv, bars_held):
        """Book a close of position ``p`` at ``raw_px`` (before sell-side slip/fee)."""
        nonlocal cash
        exit_px = raw_px * (1 - slip)
        net = p["qty"] * exit_px * (1 - fee)
        cash += net
        pnl = net - p["cost"]
        closed.append({
            "symbol": p["symbol"], "entry_date": p["entry_date"], "exit_date": pd.Timestamp(dtv),
            "entry_price": p["entry_price"], "exit_price": exit_px, "stake": p["stake"],
            "pnl": pnl, "pnl_pct": pnl / p["cost"], "bars_held": bars_held,
            "exit_reason": reason,
        })

    regime_ok = True
    for dtv, sym, i in events:
        day = pd.Timestamp(dtv).normalize()
        if day != cur_day:
            if cur_day is not None:
                mark_equity(cur_day)
            cur_day = day
            day_idx += 1
            if regime_dd is not None:
                dd_today = regime_dd.get(day)
                regime_ok = dd_today is None or dd_today > -regime_threshold

        a = prepped[sym]
        px_close = a["close"][i]
        px_open = a["open"][i]
        last_close[sym] = px_close
        if a["score"] is not None:
            last_score[sym] = a["score"][i]

        if sym in open_pos:
            p = open_pos[sym]
            raw_px = reason = None
            p_trail = p["trail"]
            if p_trail:
                p["hw"] = max(p["hw"], a["high"][i])
                if p["trail_activation_from"] == "exit_threshold" and a["exit_threshold"] is not None:
                    ref_price = a["exit_threshold"][i]
                else:
                    ref_price = p["entry_price"]
                trail_armed = p["hw"] >= ref_price * (1 + p["trail_activation"])
                trail_price = p["hw"] * (1 - p_trail) if trail_armed else -1.0
            else:
                trail_price = -1.0
            if a["low"][i] <= p["sl_price"]:
                raw_px = min(p["sl_price"], px_open) if gap else p["sl_price"]
                reason = "stoploss"
            elif p_trail and trail_price >= 0 and a["low"][i] <= trail_price and trail_price > p["entry_price"] * (1 - p["sl_pct"]):
                raw_px = min(trail_price, px_open) if gap else trail_price
                reason = "trailing_stop"
            elif a["high"][i] >= p["tp_price"]:
                raw_px, reason = p["tp_price"], "take_profit"      # resting limit fills at target
            elif a["exit"][i]:
                raw_px = px_open if next_open else px_close
                reason = "exit_signal"
            if raw_px is not None:
                _sell(p, raw_px, reason, dtv, i - p["bar"])
                del open_pos[sym]
                last_exit_day[sym] = day_idx
        elif (a["entry"][i] and _stake_for(sym, dtv) > 0 and cash >= _stake_for(sym, dtv) and regime_ok
              and day_idx - last_exit_day.get(sym, -10**9) >= risk_by_sym[sym]["cooldown"]):
            room = len(open_pos) < cfg.max_open_trades
            if not room and rotate and a["score"] is not None:
                cand_score = a["score"][i]
                if not np.isnan(cand_score):
                    # weakest-scoring open position that's past its cooldown
                    eligible = [(s, last_score.get(s, np.nan)) for s, p in open_pos.items()
                                if day_idx - p["open_day"] >= cfg.min_hold_bars
                                and not np.isnan(last_score.get(s, np.nan))]
                    if eligible:
                        weak_sym, weak_score = min(eligible, key=lambda t: t[1])
                        if cand_score - weak_score >= rotate_margin:
                            wp = open_pos[weak_sym]
                            wpx = last_close.get(weak_sym, wp["entry_price"])
                            _sell(wp, wpx, "rotated_out", dtv, day_idx - wp["open_day"])
                            del open_pos[weak_sym]
                            last_exit_day[weak_sym] = day_idx
                            room = True
            if room:
                stake = _stake_for(sym, dtv)
                ref = px_open if next_open else px_close
                entry_px = ref * (1 + slip)
                qty = stake / entry_px
                cost = stake * (1 + fee)
                cash -= cost
                r = risk_by_sym[sym]
                open_pos[sym] = {
                    "symbol": sym, "entry_date": pd.Timestamp(dtv), "entry_price": entry_px, "qty": qty,
                    "stake": stake, "cost": cost, "bar": i, "open_day": day_idx,
                    "hw": a["high"][i], "sl_price": entry_px * (1 - r["sl"]), "tp_price": entry_px * (1 + r["tp"]),
                    "sl_pct": r["sl"], "trail": r["trail"], "trail_activation": r["trail_activation"],
                    "trail_activation_from": r["trail_activation_from"],
                }

    # close survivors at last known price
    final_dt = pd.Timestamp(events[-1][0])
    for sym, p in list(open_pos.items()):
        raw_px = last_close.get(sym, p["entry_price"])
        _sell(p, raw_px, "end", final_dt, day_idx - p["open_day"])
    open_pos.clear()
    mark_equity(final_dt)

    trades = pd.DataFrame(closed, columns=[
        "symbol", "entry_date", "exit_date", "entry_price", "exit_price", "stake",
        "pnl", "pnl_pct", "bars_held", "exit_reason"])
    equity = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_dates), name="equity")
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    st = metrics.stats(equity, trades, cfg.initial_balance, cfg.timeframe)
    result_params = ({sec: dict(s.p) for sec, s in strategy.items()} if per_sector
                     else dict(strategy.p))
    return BacktestResult(trades=trades, equity=equity, stats=st, params=result_params)
