"""Agent registry. Agents self-register on import via the @register decorator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from cortex.agents.base import Agent

_REGISTRY: dict[str, type[Agent]] = {}
A = TypeVar("A", bound=type[Agent])


def register(cls: A) -> A:
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"agent name {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def available_agents() -> dict[str, str]:
    _load_builtins()
    return {name: cls.description for name, cls in sorted(_REGISTRY.items())}


def create_agent(name: str, config: Mapping[str, Any] | None = None) -> Agent:
    _load_builtins()
    try:
        cls = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"unknown agent {name!r} (known: {known})") from None
    return cls(config)


def _load_builtins() -> None:
    # Imported for their @register side effect.
    from cortex.agents import ads_optimizer, research, security_monitor  # noqa: F401
