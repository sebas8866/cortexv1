"""Memory record types.

Cortex separates three kinds of memory because they have different lifecycles:

- Episodic: what happened (append-only, decays in importance over time).
- Semantic: what is believed to be true now (versioned; new values supersede old ones,
  history is kept, every fact links back to the episodes that support it).
- Procedural: how well a way of acting has worked (success/failure counts per action kind).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class MemoryKind(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass(slots=True, frozen=True)
class Observation:
    """A claim an agent inferred. It must earn its way into semantic memory via consolidation."""

    subject: str
    predicate: str
    value: str
    confidence: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(
            subject=str(data["subject"]),
            predicate=str(data["predicate"]),
            value=str(data["value"]),
            confidence=float(data.get("confidence", 0.6)),
        )


@dataclass(slots=True)
class Episode:
    id: str
    scope: str
    agent: str
    content: str
    importance: float
    created_at: datetime
    run_id: str | None = None
    tags: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
    consolidated: bool = False

    def observations(self) -> list[Observation]:
        return [Observation.from_dict(o) for o in self.payload.get("observations", [])]


@dataclass(slots=True)
class Fact:
    id: str
    scope: str
    subject: str
    predicate: str
    value: str
    confidence: float
    evidence_count: int
    created_at: datetime
    updated_at: datetime
    source_episode_ids: list[str] = field(default_factory=list)
    superseded_by: str | None = None

    @property
    def active(self) -> bool:
        return self.superseded_by is None


@dataclass(slots=True)
class Procedure:
    id: str
    scope: str
    name: str
    description: str
    successes: int
    failures: int
    updated_at: datetime

    @property
    def success_rate(self) -> float:
        # Laplace-smoothed so a single outcome does not read as 0% or 100%.
        return (self.successes + 1) / (self.successes + self.failures + 2)


@dataclass(slots=True, frozen=True)
class RecallHit:
    kind: MemoryKind
    id: str
    scope: str
    text: str
    score: float
    relevance: float
    recency: float
    importance: float
