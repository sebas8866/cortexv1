from __future__ import annotations

import pytest

from cortex.config import ConsolidationSettings
from cortex.memory import Consolidator, MemoryKind, MemoryStore, scope_chain


def test_scope_chain_is_most_specific_first() -> None:
    assert scope_chain("acme/ads/eu") == ["acme/ads/eu", "acme/ads", "acme"]
    with pytest.raises(ValueError):
        scope_chain("/")


def test_new_value_supersedes_and_history_is_kept(memory: MemoryStore) -> None:
    first = memory.assert_fact("acme", "campaign:c1", "status", "active", confidence=1.0)
    second = memory.assert_fact("acme", "campaign:c1", "status", "paused", confidence=1.0)

    current = memory.get_fact("acme", "campaign:c1", "status")
    assert current is not None and current.id == second.id and current.value == "paused"
    history = memory.fact_history("acme", "campaign:c1", "status")
    assert [f.value for f in history] == ["active", "paused"]
    assert history[0].superseded_by == second.id
    assert first.id != second.id


def test_reinforcement_is_idempotent_per_evidence(memory: MemoryStore) -> None:
    ep = memory.record_episode("acme", "agent", "saw it", payload={})
    f1 = memory.assert_fact("acme", "s", "p", "v", confidence=0.5, sources=[ep.id])
    f2 = memory.assert_fact("acme", "s", "p", "v", confidence=0.5, sources=[ep.id])
    assert f2.confidence == pytest.approx(f1.confidence)  # same evidence, no inflation

    ep2 = memory.record_episode("acme", "agent", "saw it again")
    f3 = memory.assert_fact("acme", "s", "p", "v", confidence=0.5, sources=[ep2.id])
    assert f3.confidence == pytest.approx(0.75)
    assert f3.evidence_count == 2
    assert set(f3.source_episode_ids) == {ep.id, ep2.id}


def test_child_scope_reads_parent_but_not_reverse(memory: MemoryStore) -> None:
    memory.assert_fact("acme", "brand:acme", "tone", "plain", confidence=1.0)
    memory.assert_fact("acme/ads", "campaign:c1", "status", "active", confidence=1.0)

    assert memory.get_fact("acme/ads", "brand:acme", "tone") is not None
    assert memory.get_fact("acme", "campaign:c1", "status") is None


def test_child_fact_shadows_parent(memory: MemoryStore) -> None:
    memory.assert_fact("acme", "brand:acme", "tone", "plain", confidence=1.0)
    memory.assert_fact("acme/ads", "brand:acme", "tone", "punchy", confidence=1.0)
    facts = memory.facts_about("acme/ads", "brand:acme")
    assert [(f.predicate, f.value) for f in facts] == [("tone", "punchy")]


def test_recall_blends_relevance_recency_importance(memory: MemoryStore, clock) -> None:  # type: ignore[no-untyped-def]
    old = memory.record_episode("acme", "ads", "budget increase on brand search", importance=0.5)
    clock.advance(days=10)
    new = memory.record_episode("acme", "ads", "budget increase on brand search", importance=0.5)

    hits = memory.recall("budget brand", "acme", kinds=[MemoryKind.EPISODIC])
    assert [h.id for h in hits] == [new.id, old.id]
    assert hits[0].recency > hits[1].recency


def test_recall_skips_superseded_facts(memory: MemoryStore) -> None:
    memory.assert_fact("acme", "campaign:c1", "status", "active", confidence=1.0)
    memory.assert_fact("acme", "campaign:c1", "status", "paused", confidence=1.0)
    hits = memory.recall("campaign status", "acme", kinds=[MemoryKind.SEMANTIC])
    assert [h.text for h in hits] == ["campaign:c1 status paused"]


def test_procedure_success_rate_is_smoothed(memory: MemoryStore) -> None:
    memory.record_procedure_outcome("acme", "increase_budget", success=True)
    proc = memory.record_procedure_outcome("acme", "increase_budget", success=True)
    assert proc.successes == 2 and proc.failures == 0
    assert proc.success_rate == pytest.approx(3 / 4)


def test_consolidation_needs_corroboration(memory: MemoryStore, clock) -> None:  # type: ignore[no-untyped-def]
    obs = {
        "observations": [
            {"subject": "c1", "predicate": "tier", "value": "winner", "confidence": 0.6}
        ]
    }
    consolidator = Consolidator(memory, ConsolidationSettings(min_evidence=2, min_confidence=0.9))

    memory.record_episode("acme", "ads", "day 1", payload=obs)
    first = consolidator.run("acme")
    assert first.promoted == [] and first.deferred == 1
    assert memory.get_fact("acme", "c1", "tier") is None

    clock.advance(days=1)
    memory.record_episode("acme", "ads", "day 2", payload=obs)
    second = consolidator.run("acme")
    fact = memory.get_fact("acme", "c1", "tier")
    assert fact is not None and fact.value == "winner"
    assert fact.confidence == pytest.approx(0.84)
    assert len(fact.source_episode_ids) == 2
    assert len(second.promoted) == 1
    assert memory.unconsolidated("acme") == []


def test_unproven_observations_expire(memory: MemoryStore, clock) -> None:  # type: ignore[no-untyped-def]
    consolidator = Consolidator(memory, ConsolidationSettings(pending_ttl_days=7))
    memory.record_episode(
        "acme",
        "ads",
        "one-off",
        payload={
            "observations": [
                {"subject": "c9", "predicate": "tier", "value": "winner", "confidence": 0.3}
            ]
        },
    )
    consolidator.run("acme")
    assert len(memory.unconsolidated("acme")) == 1

    clock.advance(days=8)
    report = consolidator.run("acme")
    assert report.expired == 1
    assert memory.unconsolidated("acme") == []
    assert memory.get_fact("acme", "c9", "tier") is None
