"""Consolidation: turning repeated observations into durable beliefs.

Agents never write inferred facts directly. They attach observations to the episode
for their run, and the consolidator promotes an observation to a semantic fact only
when it is corroborated (seen in enough distinct episodes) or stated with high
confidence. Observations that never earn promotion expire after a TTL.

Things Cortex *did* (executed actions) bypass this path and are written as facts with
full confidence by the executor: an action's effect is known, not inferred.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from functools import reduce

from cortex.config import ConsolidationSettings
from cortex.memory.models import Fact
from cortex.memory.store import MemoryStore, noisy_or


@dataclass(slots=True)
class ConsolidationReport:
    scanned: int = 0
    promoted: list[Fact] = field(default_factory=list)
    deferred: int = 0
    expired: int = 0
    decayed: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "scanned": self.scanned,
            "promoted": [f"{f.subject}.{f.predicate}={f.value}" for f in self.promoted],
            "deferred": self.deferred,
            "expired": self.expired,
            "decayed": self.decayed,
        }


class Consolidator:
    def __init__(self, store: MemoryStore, settings: ConsolidationSettings | None = None) -> None:
        self.store = store
        self.settings = settings or ConsolidationSettings()

    def run(self, scope: str) -> ConsolidationReport:
        cfg = self.settings
        now = self.store.clock()
        report = ConsolidationReport()

        episodes = self.store.unconsolidated(scope)
        report.scanned = len(episodes)

        # Group identical claims across episodes: (subject, predicate, value) -> evidence.
        evidence: dict[tuple[str, str, str], list[tuple[str, float]]] = defaultdict(list)
        for ep in episodes:
            for obs in ep.observations():
                evidence[(obs.subject, obs.predicate, obs.value)].append((ep.id, obs.confidence))

        waiting: set[str] = set()
        for (subject, predicate, value), items in evidence.items():
            episode_ids = list(dict.fromkeys(eid for eid, _ in items))
            combined = reduce(noisy_or, (conf for _, conf in items), 0.0)
            # An episode can stay open for another claim; don't re-promote what it already proved.
            current = self.store.get_fact(scope, subject, predicate)
            if (
                current is not None
                and current.scope == scope
                and current.value == value
                and set(episode_ids) <= set(current.source_episode_ids)
            ):
                continue
            if len(episode_ids) >= cfg.min_evidence or combined >= cfg.min_confidence:
                fact = self.store.assert_fact(
                    scope, subject, predicate, value, confidence=combined, sources=episode_ids
                )
                report.promoted.append(fact)
            else:
                waiting.update(episode_ids)
                report.deferred += 1

        # Episodes still carrying unproven claims stay open until their TTL runs out.
        cutoff = now - timedelta(days=cfg.pending_ttl_days)
        done: list[str] = []
        for ep in episodes:
            if ep.id in waiting:
                if ep.created_at > cutoff:
                    continue
                report.expired += 1
            done.append(ep.id)
        self.store.mark_consolidated(done)

        report.decayed = self.store.decay_importance(
            now - timedelta(days=cfg.decay_after_days), cfg.decay_factor
        )
        return report
