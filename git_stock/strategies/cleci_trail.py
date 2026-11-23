"""CleciTrail — the version to hyperopt.

Buy: close comes within `(entry_mult-1)` of the trailing lowest-low over `window` bars.
Sell: EITHER close comes within `(1-exit_mult)` of the trailing highest-high,
      OR a `trail`% trailing stop off the high-water mark is hit,
      OR the `sl`% hard stop is hit.

Exactly four knobs to search: window, entry_mult, exit_mult, trail, sl.
"""
from __future__ import annotations

from .cleci import Cleci


class CleciTrail(Cleci):
    space = {
        "window": (20, 500, 200, 10),
        "entry_mult": (1.00, 1.25, 1.10, 0.01),
        "exit_mult": (0.70, 0.99, 0.90, 0.01),
        # trailing stop floored at 6% — on daily bars a 1-2% trail is just one day's
        # noise, which produced the degenerate 1-bar-hold / 4000-trade scalp configs
        "trail": (6.0, 25.0, 12.0, 0.5),
        "sl": (6.0, 30.0, 15.0, 1.0),
    }

    def __init__(self, params=None):
        fixed = {"use_rolling": True, "high_band": "high",
                 "exit_mode": "band+trail", "tp": 100.0}
        fixed.update(params or {})
        super().__init__(fixed)
