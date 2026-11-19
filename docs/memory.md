# Memory

Cortex memory is a set of record types with different lifecycles, not a single
embedding index. The question it is built to answer is *"what do we know, how sure are
we, and why?"*, not only *"what text is similar to this?"*.

## Record types

| Kind | Holds | Lifecycle |
|---|---|---|
| Episodic | What happened: one episode per agent run, per human decision. | Append-only. Importance decays after consolidation. |
| Semantic | What is believed now: `(subject, predicate) = value` with confidence. | Versioned. New value supersedes; same value reinforces. History kept. |
| Procedural | How well an action kind has worked in a scope. | Success / failure counters, Laplace-smoothed rate. |

Each fact links to the episodes that support it (`fact_sources`), so any belief can be
traced back to the runs and decisions that produced it.

## Two write paths

```
agent run ──► episode.payload.observations ──► consolidator ──► fact (if corroborated)
executor  ──► action.effects ───────────────────────────────► fact (confidence 1.0)
```

- **Observations** are inferences ("c_101 looks like a winner"). They wait in the episode
  until enough independent episodes agree (`min_evidence`) or one is confident enough
  (`min_confidence`). Unproven observations expire after `pending_ttl_days`.
- **Effects** are things Cortex did ("c_101 budget is now 180"). They are known, so they
  are written immediately.

Confidence from independent evidence combines by noisy-OR: two 0.6 observations give
0.84. Re-submitting evidence a fact has already counted does not raise its confidence.

## Scopes

Scopes are paths: `acme`, `acme/ads`, `acme/ads/eu`. Reads walk from the most specific
scope outward; writes go to exactly one scope. A child can override a parent's fact
(brand tone for one market, say) without touching it. Parents never see child memory,
which is the isolation boundary between teams or clients.

## Recall

```
score = w_rel · relevance + w_rec · recency + w_imp · importance
```

- **relevance**: BM25 from SQLite FTS5 (porter stemming), normalised to the best hit.
- **recency**: `0.5 ^ (age_hours / half_life)`, default half-life 72h.
- **importance**: episode importance, fact confidence, or procedure success rate.

Superseded facts are removed from the index, so recall only returns current beliefs.
`fact_history()` still returns the full chain.

## Planned

- Embeddings as a fourth signal (`pgvector` with the Postgres backend), keeping BM25 for
  exact identifiers such as campaign ids.
- LLM-assisted consolidation that summarises clusters of old episodes into facts while
  keeping `fact_sources` provenance.
- Contradiction policy per predicate (today: newest value wins).
