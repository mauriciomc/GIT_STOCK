"""Parameter optimisation — a small freqtrade-hyperopt-style search.

    optimise(data, StrategyCls, bt_cfg, n_trials=200, method="random", objective="sharpe")

Methods: ``random`` (default, no deps), ``grid``, ``optuna`` (TPE, if installed).
Objectives: sharpe, sortino, calmar, return, expectancy, or a "recipe" score that
blends return, Sharpe and drawdown.  Supports per-regime ``timerange`` windows so you
can optimise bull / bear / sideways slices separately (see the crypto-side recipe).
"""
from __future__ import annotations

import itertools
import random
from typing import Callable

import pandas as pd

from .backtest import BacktestConfig, run
from .strategy import Strategy


def _recipe(st: dict) -> float:
    r = (st["total_return_pct"] or 0) / 100
    sh = st["sharpe"] or 0
    dd = abs(st["max_drawdown_pct"] or 0) / 100
    return r + 0.5 * sh - 2.0 * dd


def _sharpe_sortino(st: dict) -> float:
    """Average of Sharpe and Sortino — rewards smooth, low-downside-vol equity curves
    without the return term pulling the search toward high-turnover configs."""
    sh = st["sharpe"] if st["sharpe"] is not None else -9
    so = st["sortino"] if st["sortino"] is not None else -9
    return (sh + so) / 2.0


OBJECTIVES: dict[str, Callable[[dict], float]] = {
    "sharpe": lambda s: s["sharpe"] or -9,
    "sortino": lambda s: s["sortino"] or -9,
    "sharpe_sortino": _sharpe_sortino,
    "calmar": lambda s: s["calmar"] or -9,
    "return": lambda s: s["total_return_pct"] or -9e9,
    "expectancy": lambda s: s["expectancy"] or -9e9,
    "recipe": _recipe,
}


def _sample(space: dict, rng: random.Random) -> dict:
    out = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            out[name] = rng.choice(spec)
        else:
            lo, hi = spec[0], spec[1]
            step = spec[3] if len(spec) > 3 else None
            if step and isinstance(lo, int) and isinstance(step, int):
                out[name] = rng.randrange(lo, hi + 1, step)
            elif step:
                n = int(round((hi - lo) / step))
                out[name] = round(lo + rng.randint(0, n) * step, 10)
            else:
                out[name] = rng.uniform(lo, hi)
    return out


def _grid(space: dict, cap: int) -> list[dict]:
    axes = []
    for name, spec in space.items():
        if isinstance(spec, list):
            axes.append([(name, v) for v in spec])
        else:
            lo, hi = spec[0], spec[1]
            step = spec[3] if len(spec) > 3 else (hi - lo) / 4
            vals, v = [], lo
            while v <= hi + 1e-9:
                vals.append(round(v, 10))
                v += step
            axes.append([(name, x) for x in vals])
    combos = list(itertools.product(*axes))
    if len(combos) > cap:
        combos = random.Random(0).sample(combos, cap)
    return [dict(c) for c in combos]


def optimise(data: dict[str, pd.DataFrame], strategy_cls: type[Strategy],
             bt_cfg: BacktestConfig, *, n_trials: int = 200, method: str = "random",
             objective: str = "sharpe", seed: int = 42, start=None, end=None,
             min_trades: int = 10, progress: bool = True) -> pd.DataFrame:
    score_fn = OBJECTIVES[objective]
    space = strategy_cls.space
    rng = random.Random(seed)

    if method == "grid":
        candidates = _grid(space, n_trials)
    elif method == "optuna":
        return _optuna(data, strategy_cls, bt_cfg, n_trials, objective, seed,
                       start, end, min_trades)
    else:
        candidates = [_sample(space, rng) for _ in range(n_trials)]

    rows = []
    for i, params in enumerate(candidates):
        try:
            res = run(data, strategy_cls(params), bt_cfg, start, end)
        except Exception:  # noqa: BLE001
            continue
        st = res.stats
        penalised = st["trades"] < min_trades
        rows.append({**params, **st, "score": (-9e9 if penalised else score_fn(st))})
        if progress and (i + 1) % max(1, len(candidates) // 10) == 0:
            best = max(r["score"] for r in rows)
            print(f"  {i + 1}/{len(candidates)}  best {objective}={best:.3f}")
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


# BacktestConfig fields optimise_portfolio is allowed to search — same (low, high, default,
# step) / [choices] spec format as a strategy's `space`. `rotate` is boolean-categorical so a
# trial can turn rotation off entirely and search the rest as a no-rotation baseline.
PORTFOLIO_SPACE: dict = {
    "stake_amount": (10.0, 200.0, 40.0, 5.0),
    "max_open_trades": (5, 60, 40, 1),
    "rotate": [False, True],
    "rotate_margin_pct": (0.5, 10.0, 1.0, 0.5),
    "min_hold_bars": (0, 30, 5, 1),
    # 0 = off (legacy); e.g. 15.0 = pause new entries while the market-proxy drawdown
    # (bt_cfg.regime_dd, set once by the caller) is worse than -15%. 0 stays in-range so
    # hyperopt can find "no filter helps" on its own rather than it being forced.
    "regime_dd_threshold": (0.0, 30.0, 0.0, 1.0),
}


def optimise_portfolio(data: dict[str, pd.DataFrame], strategy_cls: type[Strategy],
                       bt_cfg: BacktestConfig, *, bt_space: dict = PORTFOLIO_SPACE,
                       n_trials: int = 500, objective: str = "sharpe", seed: int = 42,
                       start=None, end=None, min_trades: int = 10,
                       progress: bool = True) -> pd.DataFrame:
    """Like ``optimise``, but also searches ``bt_space`` (BacktestConfig fields — stake
    size, position count, rotation) jointly with the strategy's own ``space``, instead of
    holding them fixed from config.json. Each trial builds its own BacktestConfig via
    ``dataclasses.replace(bt_cfg, **bt_overrides)``, so bt_cfg's other fields (cost model
    etc.) stay whatever the caller set."""
    import dataclasses

    score_fn = OBJECTIVES[objective]
    strat_names = list(strategy_cls.space)
    bt_names = list(bt_space)
    combined = {**strategy_cls.space, **bt_space}
    rng = random.Random(seed)
    candidates = [_sample(combined, rng) for _ in range(n_trials)]

    rows = []
    for i, full in enumerate(candidates):
        strat_params = {k: full[k] for k in strat_names}
        bt_overrides = {k: full[k] for k in bt_names}
        # rotate=False makes rotate_margin_pct/min_hold_bars moot — don't waste trials
        # exploring their combinations when rotation is off (dedupe isn't needed for
        # random search quality, but this keeps them harmlessly present in `full` for
        # the results table regardless).
        try:
            trial_cfg = dataclasses.replace(bt_cfg, **bt_overrides)
            res = run(data, strategy_cls(strat_params), trial_cfg, start, end)
        except Exception:  # noqa: BLE001
            continue
        st = res.stats
        penalised = st["trades"] < min_trades
        rows.append({**full, **st, "score": (-9e9 if penalised else score_fn(st))})
        if progress and (i + 1) % max(1, len(candidates) // 20) == 0:
            best = max(r["score"] for r in rows)
            print(f"  {i + 1}/{len(candidates)}  best {objective}={best:.3f}")
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


def _optuna(data, strategy_cls, bt_cfg, n_trials, objective, seed, start, end, min_trades):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    score_fn = OBJECTIVES[objective]
    space = strategy_cls.space
    rows: list[dict] = []

    def obj(trial: "optuna.Trial") -> float:
        params = {}
        for name, spec in space.items():
            if isinstance(spec, list):
                params[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec[0], int) and (len(spec) < 4 or isinstance(spec[3], int)):
                params[name] = trial.suggest_int(name, spec[0], spec[1], step=spec[3] if len(spec) > 3 else 1)
            else:
                params[name] = trial.suggest_float(name, spec[0], spec[1], step=spec[3] if len(spec) > 3 else None)
        res = run(data, strategy_cls(params), bt_cfg, start, end)
        st = res.stats
        rows.append({**params, **st})
        return -9e9 if st["trades"] < min_trades else score_fn(st)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    df = pd.DataFrame(rows)
    df["score"] = [(-9e9 if r["trades"] < min_trades else score_fn(r)) for r in rows]
    return df.sort_values("score", ascending=False).reset_index(drop=True)
