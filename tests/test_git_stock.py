"""Deterministic tests on synthetic data — no network."""
import numpy as np
import pandas as pd
import pytest

from git_stock import indicators as ind
from git_stock.backtest import BacktestConfig, run
from git_stock.strategies import available, get
from git_stock.strategy import Strategy, actionable_signals


def _series(vals):
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="D")
    return pd.Series(vals, index=idx, dtype=float)


def _ohlcv(closes):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    c = np.array(closes, float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "adj_close": c, "volume": 1e6}, index=idx)


# ---- indicators ------------------------------------------------------
def test_crossed_above_below():
    a = _series([1, 2, 3, 2, 1])
    b = _series([2, 2, 2, 2, 2])
    assert list(ind.crossed_above(a, b)) == [False, False, True, False, False]
    assert list(ind.crossed_below(a, b)) == [False, False, False, False, True]


def test_crossed_scalar():
    a = _series([9, 11, 9])
    assert list(ind.crossed_above(a, 10)) == [False, True, False]


def test_rsi_bounds():
    r = ind.rsi(_series(np.linspace(10, 50, 60)))
    r = r.dropna()
    assert (r >= 0).all() and (r <= 100).all()
    assert r.iloc[-1] > 90  # pure uptrend


def test_rolling_min_max():
    s = _series([5, 3, 4, 2, 6])
    assert list(ind.rolling_min(s, 2)) == [5, 3, 3, 2, 2]
    assert list(ind.rolling_max(s, 2)) == [5, 5, 4, 4, 6]


# ---- backtester ----------------------------------------------------
class _AlwaysFlat(Strategy):
    space = {}

    def populate_indicators(self, df):
        return df

    def entry_signal(self, df):
        return pd.Series(False, index=df.index)

    def exit_signal(self, df):
        return pd.Series(False, index=df.index)


class _OneRoundTrip(Strategy):
    """Enter on bar 2, exit on bar 6."""
    space = {}

    def populate_indicators(self, df):
        return df

    def entry_signal(self, df):
        s = pd.Series(False, index=df.index)
        s.iloc[2] = True
        return s

    def exit_signal(self, df):
        s = pd.Series(False, index=df.index)
        s.iloc[6] = True
        return s


def test_backtest_no_trades():
    data = {"AAA": _ohlcv([10] * 20)}
    res = run(data, _AlwaysFlat(), BacktestConfig(initial_balance=1000))
    assert res.stats["trades"] == 0
    assert res.stats["end_balance"] == pytest.approx(1000, abs=1e-6)


def test_backtest_single_winning_trade():
    prices = [10, 10, 10, 11, 12, 13, 14, 14, 14, 14]  # enter@10 (bar2), exit@14 (bar6)
    data = {"AAA": _ohlcv(prices)}
    cfg = BacktestConfig(initial_balance=1000, stake_amount=100, fee=0.0,
                         stoploss_pct=90, take_profit_pct=1000)
    res = run(data, _OneRoundTrip(), cfg)
    assert res.stats["trades"] == 1
    t = res.trades.iloc[0]
    assert t["entry_price"] == 10 and t["exit_price"] == 14
    assert t["pnl"] == pytest.approx(40.0, abs=1e-6)      # 100 -> 140
    assert res.stats["end_balance"] == pytest.approx(1040.0, abs=1e-6)


def test_backtest_stoploss_fires():
    prices = [10, 10, 10, 9, 8, 7, 7, 7, 7, 7]
    data = {"AAA": _ohlcv(prices)}
    cfg = BacktestConfig(initial_balance=1000, stake_amount=100, fee=0.0,
                         stoploss_pct=10, take_profit_pct=1000)
    res = run(data, _OneRoundTrip(), cfg)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "stoploss"
    assert t["exit_price"] == pytest.approx(9.0)          # 10 * (1 - 0.10)


def test_max_open_trades_cap():
    data = {s: _ohlcv([10, 10, 10, 11, 12, 13, 14, 14, 14, 14]) for s in ("A", "B", "C")}
    cfg = BacktestConfig(initial_balance=1000, stake_amount=100, max_open_trades=2, fee=0.0)
    res = run(data, _OneRoundTrip(), cfg)
    assert res.stats["trades"] == 2                       # third entry blocked


# ---- signal de-duplication --------------------------------------
def test_actionable_signals_dedupe():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    #                b   b   b   .   s   s   .   b   .   s
    entry = pd.Series([1, 1, 1, 0, 0, 0, 0, 1, 0, 0], idx, dtype=bool)
    exit_ = pd.Series([0, 0, 0, 0, 1, 1, 0, 0, 0, 1], idx, dtype=bool)
    ae, ax = actionable_signals(entry, exit_)
    assert list(ae) == [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]   # only the first buy of each run
    assert list(ax) == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]   # only the first sell while long
    assert ae.sum() == ax.sum() == 2                     # complete round trips


def test_actionable_signals_exit_before_entry_ignored():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    entry = pd.Series([0, 0, 1, 0, 0], idx, dtype=bool)
    exit_ = pd.Series([1, 1, 0, 0, 1], idx, dtype=bool)
    ae, ax = actionable_signals(entry, exit_)
    assert list(ae) == [0, 0, 1, 0, 0]
    assert list(ax) == [0, 0, 0, 0, 1]                   # exits while flat are dropped


# ---- friction model ----------------------------------------------
def test_costs_reduce_pnl():
    """fee + slippage + half-spread must all bite a round trip."""
    prices = [10, 10, 10, 11, 12, 13, 14, 14, 14, 14]
    data = {"AAA": _ohlcv(prices)}
    base = dict(initial_balance=1000, stake_amount=100, stoploss_pct=90, take_profit_pct=1000)
    clean = run(data, _OneRoundTrip(), BacktestConfig(fee=0.0, **base)).stats["end_balance"]
    fee = run(data, _OneRoundTrip(), BacktestConfig(fee=0.003, **base)).stats["end_balance"]
    dirty = run(data, _OneRoundTrip(),
                BacktestConfig(fee=0.003, slippage_pct=0.002, spread_pct=0.004, **base)
                ).stats["end_balance"]
    assert clean > fee > dirty            # each extra cost lowers the ending balance
    # buy ~10, sell ~14: round-trip slip+spread ≈ 2*(0.002+0.002) on ~40% gross move
    assert clean - dirty > 1.0


def test_fill_next_open_uses_next_bar_open():
    # entry signal on bar 2 -> with next-open it fills at bar 3's OPEN, not bar 2's close
    closes = [10, 10, 10, 20, 20, 20, 25, 25, 25, 25]
    df = _ohlcv(closes)
    df.loc[df.index[3], "open"] = 18.0            # a distinct, known open on the fill bar
    cfg = BacktestConfig(initial_balance=1000, stake_amount=100, fee=0.0,
                         stoploss_pct=90, take_profit_pct=1000, fill_next_open=True)
    t = run({"AAA": df}, _OneRoundTrip(), cfg).trades.iloc[0]
    assert t["entry_price"] == pytest.approx(18.0)


def test_gap_stops_fill_below_the_stop():
    # price gaps straight through a 10% stop on bar 3 (opens at 8.0, stop is 9.0)
    closes = [10, 10, 10, 8, 8, 8, 8, 8, 8, 8]
    df = _ohlcv(closes)
    df.loc[df.index[3], "open"] = 8.0
    base = dict(initial_balance=1000, stake_amount=100, fee=0.0, stoploss_pct=10, take_profit_pct=1000)
    no_gap = run({"AAA": df}, _OneRoundTrip(), BacktestConfig(**base)).trades.iloc[0]
    gapped = run({"AAA": df}, _OneRoundTrip(), BacktestConfig(gap_stops=True, **base)).trades.iloc[0]
    assert no_gap["exit_price"] == pytest.approx(9.0)     # optimistic: fills at the stop
    assert gapped["exit_price"] == pytest.approx(8.0)     # realistic: fills at the gap open


# ---- strategies load + fixed cleci --------------------------------
def test_strategies_registry():
    assert {"cleci", "emacross", "bbrsi"} <= set(available())


def _wobble(scale, seed):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.02, 200)
    return _ohlcv(list(scale * np.exp(np.cumsum(steps))))


def test_cleci_is_stateless_across_calls():
    """The old bug: highest_low / lowest_low were class attributes that leaked between
    symbols, so only the first ticker in a run could signal. Each prepare() is now
    fully independent and reproducible."""
    cleci = get("cleci")({"window": 30})
    a = _wobble(100, 1)          # big price scale
    b = _wobble(3, 2)            # tiny price scale, different symbol

    ra1 = cleci.prepare(a)
    _ = cleci.prepare(b)        # would have polluted shared state in the old version
    ra2 = cleci.prepare(a)
    pd.testing.assert_series_equal(ra1["entry"], ra2["entry"])   # idempotent
    pd.testing.assert_series_equal(ra1["band_low"], ra2["band_low"])

    # both symbols independently produce a mix of entry signals
    assert 0 < cleci.prepare(a)["entry"].sum() < len(a)
    assert 0 < cleci.prepare(b)["entry"].sum() < len(b)
