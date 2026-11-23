"""Strategy registry — `get(name)` returns the class."""
from __future__ import annotations

import importlib
import pkgutil

from ..strategy import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}


def _discover() -> None:
    if _REGISTRY:
        return
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        m = importlib.import_module(f"{__name__}.{mod.name}")
        for obj in vars(m).values():
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                _REGISTRY[obj.__name__.lower()] = obj


def get(name: str) -> type[Strategy]:
    _discover()
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; have {sorted(_REGISTRY)}")


def available() -> list[str]:
    _discover()
    return sorted(_REGISTRY)
