"""Typed configuration loaded from YAML with a small set of environment overrides.

Precedence: explicit path > $CORTEX_CONFIG > ./config/cortex.yaml > built-in defaults.
Secrets (API keys) are never read from YAML; they come from the environment only.
"""

from __future__ import annotations

import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    provider: Literal["offline", "anthropic"] = "offline"
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 800


class RecallWeights(BaseModel):
    """How recall blends the three signals. Should sum to ~1.0."""

    relevance: float = 0.5
    recency: float = 0.25
    importance: float = 0.25


class ConsolidationSettings(BaseModel):
    min_evidence: int = 2  # distinct episodes needed to promote an observation to a fact
    min_confidence: float = 0.75  # ...or a single observation this confident
    pending_ttl_days: int = 7  # unpromoted observations are dropped after this
    decay_after_days: int = 14  # consolidated episodes start losing importance after this
    decay_factor: float = 0.5


class MemorySettings(BaseModel):
    recall_weights: RecallWeights = Field(default_factory=RecallWeights)
    recency_half_life_hours: float = 72.0
    consolidation: ConsolidationSettings = Field(default_factory=ConsolidationSettings)


class PolicySettings(BaseModel):
    max_auto_risk: Literal["low", "medium", "high"] = "low"
    max_budget_change_pct: float = 25.0
    daily_auto_action_limit: int = 20
    denied_kinds: list[str] = Field(default_factory=list)
    dry_run: bool = True


class QueueSettings(BaseModel):
    backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    name: str = "cortex:tasks"


class Schedule(BaseModel):
    agent: str
    every: str
    scope: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)

    @property
    def interval(self) -> timedelta:
        return parse_interval(self.every)


class Settings(BaseModel):
    data_dir: Path = Path("var")
    default_scope: str = "personal"
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    schedules: list[Schedule] = Field(default_factory=list)
    agents: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "cortex.db"


_INTERVAL = re.compile(r"^\s*(\d+)\s*([smhd])\s*$")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def parse_interval(spec: str) -> timedelta:
    """Parse '30s', '15m', '6h', '1d' into a timedelta."""
    match = _INTERVAL.match(spec)
    if not match:
        raise ValueError(f"invalid interval {spec!r}; expected e.g. '15m', '6h', '1d'")
    amount, unit = int(match.group(1)), match.group(2)
    if amount <= 0:
        raise ValueError(f"interval must be positive: {spec!r}")
    return timedelta(**{_UNITS[unit]: amount})


def load_settings(path: str | Path | None = None) -> Settings:
    candidate = Path(path or os.environ.get("CORTEX_CONFIG", "config/cortex.yaml"))
    raw: dict[str, Any] = {}
    if candidate.exists():
        raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    settings = Settings.model_validate(raw)

    if provider := os.environ.get("CORTEX_LLM_PROVIDER"):
        settings.llm.provider = provider  # type: ignore[assignment]
    if data_dir := os.environ.get("CORTEX_DATA_DIR"):
        settings.data_dir = Path(data_dir)
    if redis_url := os.environ.get("CORTEX_REDIS_URL"):
        settings.queue.redis_url = redis_url
    if backend := os.environ.get("CORTEX_QUEUE_BACKEND"):
        settings.queue.backend = backend  # type: ignore[assignment]
    # Re-validate so a bad override (e.g. provider "gpt") fails here, not deep in a run.
    return Settings.model_validate(settings.model_dump())
