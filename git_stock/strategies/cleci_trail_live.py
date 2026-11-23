"""CleciTrailLive — fixed reference config, NOT hyperopted.

Pins the exact `clecitrail` parameters currently sitting in `config/config.json`
(the config the telegram bot / signals runner would load), so the walk-forward
bake-off can score that specific configuration head-to-head against the freshly
hyperopted `clecitrail` / `cleci` / etc. winners, instead of just trusting the
note left in config.json's `_strategy_note`.

Those params come from bake-off **Run 1** (frictionless, loose bounds — see
bt/RESULTS.md), which the bake-off's own writeup flagged as a probable artifact:
a sub-6% trailing stop that just scalps one day's noise on daily bars. `space`
is intentionally empty here so `optimise()` can't accidentally search around it;
this class exists only to be instantiated directly with its fixed defaults.
"""
from __future__ import annotations

from .cleci_trail import CleciTrail

# Exactly config/config.json's current "strategy_params" for "strategy": "clecitrail".
LIVE_PARAMS = {
    "window": 120,
    "entry_mult": 1.16,
    "exit_mult": 0.77,
    "trail": 2.0,
    "sl": 29.0,
}


class CleciTrailLive(CleciTrail):
    space: dict = {}  # fixed config — nothing to hyperopt

    def __init__(self, params=None):
        fixed = dict(LIVE_PARAMS)
        fixed.update(params or {})
        super().__init__(fixed)
