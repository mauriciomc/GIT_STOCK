"""Typed configuration, loaded from the existing config/*.json schema."""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class Config:
    user: str = "default"
    strategy: str = "cleci"
    timeframe: str = "1d"
    daterange: int = 2000            # bars of history to pull
    timewindow: int = 200           # strategy lookback (legacy name)

    initial_balance: float = 1000.0
    stake_amount: float = 40.0
    max_open_trades: int = 40
    tradable_balance_ratio: float = 0.99

    open_fee: float = 0.001
    close_fee: float = 0.001
    stop_loss: float = 5.0           # percent
    take_profit: float = 100.0       # percent

    database: str = "signals.db"
    strategy_params: dict[str, Any] = field(default_factory=dict)  # extra params for the strategy
    sectors: dict[str, list[str]] = field(default_factory=dict)
    telegram: dict[str, Any] = field(default_factory=dict)

    # ---- helpers -------------------------------------------------------
    @property
    def tickers(self) -> list[str]:
        seen: dict[str, None] = {}
        for names in self.sectors.values():
            for t in names:
                seen.setdefault(t, None)
        return list(seen)

    def sector_of(self, ticker: str) -> str | None:
        for sector, names in self.sectors.items():
            if ticker in names:
                return sector
        return None

    @classmethod
    def load(cls, path: str | pathlib.Path, secrets: str | pathlib.Path | None = None) -> "Config":
        """Load the shareable config, then overlay private values.

        Secrets never live in config.json (it is committed). They come from, in order:
        1. `secrets.json` next to the config (git-ignored; copy secrets.example.json),
           or the path given as `secrets` / $GIT_STOCK_SECRETS;
        2. env vars GIT_STOCK_TELEGRAM_TOKEN and GIT_STOCK_TELEGRAM_CHAT_ID (win over the file).
        """
        path = pathlib.Path(path)
        raw = json.loads(path.read_text())
        sp = pathlib.Path(secrets or os.environ.get("GIT_STOCK_SECRETS") or path.with_name("secrets.json"))
        if sp.is_file():
            _merge(raw, json.loads(sp.read_text()))
        tg = raw.setdefault("telegram", {})
        for key, env in (("token", "GIT_STOCK_TELEGRAM_TOKEN"), ("chat_id", "GIT_STOCK_TELEGRAM_CHAT_ID")):
            if os.environ.get(env):
                tg[key] = os.environ[env]
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def _merge(base: dict, extra: dict) -> None:
    """Recursively overlay `extra` onto `base` in place (dicts merge, other values replace)."""
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
