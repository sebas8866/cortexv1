"""SQLite-backed memory store with FTS5 retrieval.

Design notes
------------
* Scopes are hierarchical, separated by '/': ``acme/ads/eu`` can read memories written
  at ``acme/ads`` and ``acme``, never the other way round. This is how one install serves
  a single person and a multi-team org with the same code.
* Facts are versioned per (scope, subject, predicate). A different value supersedes the
  old fact (history is kept); the same value reinforces it (noisy-OR on confidence), but
  only when backed by evidence the fact has not already counted.
* Recall blends lexical relevance (BM25), recency (exponential half-life) and importance
  (episode importance, fact confidence, procedure success rate). Embeddings can be added
  as a fourth signal without changing callers; see docs/memory.md.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from cortex.config import MemorySettings
from cortex.db import Database, from_iso, new_id, to_iso, utcnow
from cortex.memory.models import Episode, Fact, MemoryKind, Procedure, RecallHit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id           TEXT PRIMARY KEY,
    scope        TEXT NOT NULL,
    agent        TEXT NOT NULL,
    run_id       TEXT,
    content      TEXT NOT NULL,
    payload      TEXT NOT NULL DEFAULT '{}',
    tags         TEXT NOT NULL DEFAULT '',
    importance   REAL NOT NULL,
    created_at   TEXT NOT NULL,
    consolidated INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS episodes_scope_time ON episodes (scope, created_at);
CREATE INDEX IF NOT EXISTS episodes_pending ON episodes (scope, consolidated);

CREATE TABLE IF NOT EXISTS facts (
    id             TEXT PRIMARY KEY,
    scope          TEXT NOT NULL,
    subject        TEXT NOT NULL,
    predicate      TEXT NOT NULL,
    value          TEXT NOT NULL,
    confidence     REAL NOT NULL,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    superseded_by  TEXT REFERENCES facts (id)
);
CREATE INDEX IF NOT EXISTS facts_lookup ON facts (scope, subject, predicate);
CREATE UNIQUE INDEX IF NOT EXISTS facts_one_active
    ON facts (scope, subject, predicate) WHERE superseded_by IS NULL;

CREATE TABLE IF NOT EXISTS fact_sources (
    fact_id    TEXT NOT NULL REFERENCES facts (id),
    episode_id TEXT NOT NULL REFERENCES episodes (id),
    PRIMARY KEY (fact_id, episode_id)
);

CREATE TABLE IF NOT EXISTS procedures (
    id          TEXT PRIMARY KEY,
    scope       TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    successes   INTEGER NOT NULL DEFAULT 0,
    failures    INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL,
    UNIQUE (scope, name)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5 (
    ref_id UNINDEXED,
    kind   UNINDEXED,
    scope  UNINDEXED,
    text,
    tokenize = 'porter unicode61'
);
"""

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def scope_chain(scope: str) -> list[str]:
    """'acme/ads/eu' -> ['acme/ads/eu', 'acme/ads', 'acme'] (most specific first)."""
    parts = [p for p in scope.strip("/").split("/") if p]
    if not parts:
        raise ValueError("scope must be a non-empty path like 'personal' or 'acme/ads'")
    return ["/".join(parts[:i]) for i in range(len(parts), 0, -1)]


def noisy_or(a: float, b: float) -> float:
    """Combine two independent pieces of evidence for the same claim."""
    return min(1.0, 1.0 - (1.0 - a) * (1.0 - b))


class MemoryStore:
    def __init__(
        self,
        path: str | Path,
        *,
        settings: MemorySettings | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._db = Database(path, _SCHEMA)
        self._settings = settings or MemorySettings()
        self.clock = clock

    # ------------------------------------------------------------------ episodic

    def record_episode(
        self,
        scope: str,
        agent: str,
        content: str,
        *,
        importance: float = 0.5,
        run_id: str | None = None,
        tags: Iterable[str] = (),
        payload: dict[str, Any] | None = None,
    ) -> Episode:
        scope_chain(scope)  # validate
        episode = Episode(
            id=new_id("ep"),
            scope=scope,
            agent=agent,
            content=content,
            importance=max(0.0, min(1.0, importance)),
            created_at=self.clock(),
            run_id=run_id,
            tags=tuple(tags),
            payload=payload or {},
        )
        with self._db.tx() as c:
            c.execute(
                "INSERT INTO episodes (id, scope, agent, run_id, content, payload, tags,"
                " importance, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    episode.id,
                    scope,
                    agent,
                    run_id,
                    content,
                    json.dumps(episode.payload, default=str),
                    ",".join(episode.tags),
                    episode.importance,
                    to_iso(episode.created_at),
                ),
            )
            self._index(c, episode.id, MemoryKind.EPISODIC, scope, f"{agent} {content}")
        return episode

    def episode(self, episode_id: str) -> Episode | None:
        rows = self._db.query("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        return _episode(rows[0]) if rows else None

    def recent_episodes(self, scope: str, limit: int = 20) -> list[Episode]:
        chain = scope_chain(scope)
        rows = self._db.query(
            f"SELECT * FROM episodes WHERE scope IN ({_marks(chain)})"
            " ORDER BY created_at DESC LIMIT ?",
            [*chain, limit],
        )
        return [_episode(r) for r in rows]

    def unconsolidated(self, scope: str, limit: int = 500) -> list[Episode]:
        rows = self._db.query(
            "SELECT * FROM episodes WHERE scope = ? AND consolidated = 0"
            " ORDER BY created_at LIMIT ?",
            (scope, limit),
        )
        return [_episode(r) for r in rows]

    def mark_consolidated(self, episode_ids: Sequence[str]) -> None:
        if not episode_ids:
            return
        with self._db.tx() as c:
            c.executemany(
                "UPDATE episodes SET consolidated = 1 WHERE id = ?", [(i,) for i in episode_ids]
            )

    def decay_importance(self, older_than: datetime, factor: float) -> int:
        """Fade consolidated episodes so recall favours what still matters."""
        with self._db.tx() as c:
            cur = c.execute(
                "UPDATE episodes SET importance = importance * ?"
                " WHERE consolidated = 1 AND created_at < ? AND importance > 0.05",
                (factor, to_iso(older_than)),
            )
            return cur.rowcount

    # ------------------------------------------------------------------ semantic

    def assert_fact(
        self,
        scope: str,
        subject: str,
        predicate: str,
        value: str,
        *,
        confidence: float,
        sources: Iterable[str] = (),
    ) -> Fact:
        scope_chain(scope)
        now = to_iso(self.clock())
        sources = list(dict.fromkeys(sources))
        with self._db.tx() as c:
            current = c.execute(
                "SELECT * FROM facts WHERE scope = ? AND subject = ? AND predicate = ?"
                " AND superseded_by IS NULL",
                (scope, subject, predicate),
            ).fetchone()

            if current is not None and current["value"] == value:
                fact_id = current["id"]
                known = {
                    r["episode_id"]
                    for r in c.execute(
                        "SELECT episode_id FROM fact_sources WHERE fact_id = ?", (fact_id,)
                    )
                }
                fresh = [s for s in sources if s not in known]
                # Re-asserting with already-counted evidence must not inflate confidence.
                if fresh or not sources:
                    c.execute(
                        "UPDATE facts SET confidence = ?, evidence_count = evidence_count + 1,"
                        " updated_at = ? WHERE id = ?",
                        (noisy_or(current["confidence"], confidence), now, fact_id),
                    )
                sources = fresh
            else:
                fact_id = new_id("fact")
                if current is not None:
                    # Release the one-active-fact slot before inserting the replacement;
                    # the FK to the not-yet-inserted fact is checked at commit.
                    c.execute("PRAGMA defer_foreign_keys = ON")
                    c.execute(
                        "UPDATE facts SET superseded_by = ? WHERE id = ?", (fact_id, current["id"])
                    )
                    c.execute("DELETE FROM memory_fts WHERE ref_id = ?", (current["id"],))
                c.execute(
                    "INSERT INTO facts (id, scope, subject, predicate, value, confidence,"
                    " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (fact_id, scope, subject, predicate, value, confidence, now, now),
                )
                self._index(
                    c, fact_id, MemoryKind.SEMANTIC, scope, f"{subject} {predicate} {value}"
                )

            c.executemany(
                "INSERT OR IGNORE INTO fact_sources (fact_id, episode_id) VALUES (?, ?)",
                [(fact_id, s) for s in sources],
            )
        fact = self._fact_by_id(fact_id)
        assert fact is not None
        return fact

    def get_fact(self, scope: str, subject: str, predicate: str) -> Fact | None:
        """The active fact, resolved from the most specific scope outward."""
        for s in scope_chain(scope):
            rows = self._db.query(
                "SELECT id FROM facts WHERE scope = ? AND subject = ? AND predicate = ?"
                " AND superseded_by IS NULL",
                (s, subject, predicate),
            )
            if rows:
                return self._fact_by_id(rows[0]["id"])
        return None

    def facts_about(self, scope: str, subject: str) -> list[Fact]:
        chain = scope_chain(scope)
        rows = self._db.query(
            f"SELECT id, scope, predicate FROM facts WHERE subject = ? AND superseded_by IS NULL"
            f" AND scope IN ({_marks(chain)})",
            [subject, *chain],
        )
        # A child scope's fact shadows the same predicate in a parent scope.
        rank = {s: i for i, s in enumerate(chain)}
        best: dict[str, sqlite3.Row] = {}
        for r in sorted(rows, key=lambda r: rank[r["scope"]]):
            best.setdefault(r["predicate"], r)
        facts = [self._fact_by_id(r["id"]) for r in best.values()]
        return sorted((f for f in facts if f), key=lambda f: f.predicate)

    def fact_history(self, scope: str, subject: str, predicate: str) -> list[Fact]:
        rows = self._db.query(
            "SELECT id FROM facts WHERE scope = ? AND subject = ? AND predicate = ?"
            " ORDER BY created_at",
            (scope, subject, predicate),
        )
        return [f for r in rows if (f := self._fact_by_id(r["id"]))]

    def _fact_by_id(self, fact_id: str) -> Fact | None:
        rows = self._db.query("SELECT * FROM facts WHERE id = ?", (fact_id,))
        if not rows:
            return None
        r = rows[0]
        sources = self._db.query(
            "SELECT episode_id FROM fact_sources WHERE fact_id = ? ORDER BY episode_id", (fact_id,)
        )
        return Fact(
            id=r["id"],
            scope=r["scope"],
            subject=r["subject"],
            predicate=r["predicate"],
            value=r["value"],
            confidence=r["confidence"],
            evidence_count=r["evidence_count"],
            created_at=from_iso(r["created_at"]),
            updated_at=from_iso(r["updated_at"]),
            source_episode_ids=[s["episode_id"] for s in sources],
            superseded_by=r["superseded_by"],
        )

    # ---------------------------------------------------------------- procedural

    def record_procedure_outcome(
        self, scope: str, name: str, *, success: bool, description: str = ""
    ) -> Procedure:
        now = to_iso(self.clock())
        with self._db.tx() as c:
            row = c.execute(
                "SELECT id FROM procedures WHERE scope = ? AND name = ?", (scope, name)
            ).fetchone()
            if row is None:
                proc_id = new_id("proc")
                c.execute(
                    "INSERT INTO procedures (id, scope, name, description, successes, failures,"
                    " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (proc_id, scope, name, description, int(success), int(not success), now),
                )
                self._index(c, proc_id, MemoryKind.PROCEDURAL, scope, f"{name} {description}")
            else:
                proc_id = row["id"]
                column = "successes" if success else "failures"
                c.execute(
                    f"UPDATE procedures SET {column} = {column} + 1, updated_at = ? WHERE id = ?",
                    (now, proc_id),
                )
        proc = self.procedure(scope, name)
        assert proc is not None
        return proc

    def procedure(self, scope: str, name: str) -> Procedure | None:
        rows = self._db.query(
            "SELECT * FROM procedures WHERE scope = ? AND name = ?", (scope, name)
        )
        return _procedure(rows[0]) if rows else None

    # -------------------------------------------------------------------- recall

    def recall(
        self,
        query: str,
        scope: str,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int = 8,
    ) -> list[RecallHit]:
        terms = _TOKEN.findall(query.lower())
        if not terms:
            return []
        chain = scope_chain(scope)
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
        sql = (
            "SELECT ref_id, kind, scope, text, bm25(memory_fts) AS rank FROM memory_fts"
            f" WHERE memory_fts MATCH ? AND scope IN ({_marks(chain)})"
        )
        params: list[object] = [match, *chain]
        if kinds is not None:
            kind_list = [k.value for k in kinds]
            sql += f" AND kind IN ({_marks(kind_list)})"
            params.extend(kind_list)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit * 5)  # over-fetch, then re-rank with recency and importance
        rows = self._db.query(sql, params)
        if not rows:
            return []

        # bm25() is lower-is-better and negative; flip and normalise to [0, 1].
        raw = {r["ref_id"]: max(0.0, -float(r["rank"])) for r in rows}
        top = max(raw.values()) or 1.0
        now = self.clock()
        w = self._settings.recall_weights
        half_life = self._settings.recency_half_life_hours

        hits: list[RecallHit] = []
        for r in rows:
            meta = self._recall_meta(MemoryKind(r["kind"]), r["ref_id"])
            if meta is None:
                continue
            importance, ts = meta
            age_h = max(0.0, (now - ts).total_seconds() / 3600)
            recency = math.exp(-math.log(2) * age_h / half_life)
            relevance = raw[r["ref_id"]] / top
            score = w.relevance * relevance + w.recency * recency + w.importance * importance
            hits.append(
                RecallHit(
                    kind=MemoryKind(r["kind"]),
                    id=r["ref_id"],
                    scope=r["scope"],
                    text=r["text"],
                    score=round(score, 4),
                    relevance=round(relevance, 4),
                    recency=round(recency, 4),
                    importance=round(importance, 4),
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def _recall_meta(self, kind: MemoryKind, ref_id: str) -> tuple[float, datetime] | None:
        if kind is MemoryKind.EPISODIC:
            rows = self._db.query(
                "SELECT importance, created_at FROM episodes WHERE id = ?", (ref_id,)
            )
            return (rows[0]["importance"], from_iso(rows[0]["created_at"])) if rows else None
        if kind is MemoryKind.SEMANTIC:
            rows = self._db.query(
                "SELECT confidence, updated_at FROM facts WHERE id = ? AND superseded_by IS NULL",
                (ref_id,),
            )
            return (rows[0]["confidence"], from_iso(rows[0]["updated_at"])) if rows else None
        rows = self._db.query("SELECT * FROM procedures WHERE id = ?", (ref_id,))
        if not rows:
            return None
        proc = _procedure(rows[0])
        return proc.success_rate, proc.updated_at

    # --------------------------------------------------------------------- misc

    def stats(self, scope: str) -> dict[str, int]:
        chain = scope_chain(scope)
        marks = _marks(chain)
        q = self._db.query
        return {
            "episodes": q(f"SELECT COUNT(*) n FROM episodes WHERE scope IN ({marks})", chain)[0][
                "n"
            ],
            "episodes_pending": q(
                f"SELECT COUNT(*) n FROM episodes WHERE consolidated = 0 AND scope IN ({marks})",
                chain,
            )[0]["n"],
            "facts_active": q(
                f"SELECT COUNT(*) n FROM facts WHERE superseded_by IS NULL AND scope IN ({marks})",
                chain,
            )[0]["n"],
            "facts_superseded": q(
                f"SELECT COUNT(*) n FROM facts WHERE superseded_by IS NOT NULL"
                f" AND scope IN ({marks})",
                chain,
            )[0]["n"],
            "procedures": q(f"SELECT COUNT(*) n FROM procedures WHERE scope IN ({marks})", chain)[
                0
            ]["n"],
        }

    def close(self) -> None:
        self._db.close()

    @staticmethod
    def _index(c: sqlite3.Connection, ref_id: str, kind: MemoryKind, scope: str, text: str) -> None:
        c.execute(
            "INSERT INTO memory_fts (ref_id, kind, scope, text) VALUES (?, ?, ?, ?)",
            (ref_id, kind.value, scope, text),
        )


def _marks(items: Sequence[object]) -> str:
    return ", ".join("?" for _ in items)


def _episode(r: sqlite3.Row) -> Episode:
    return Episode(
        id=r["id"],
        scope=r["scope"],
        agent=r["agent"],
        content=r["content"],
        importance=r["importance"],
        created_at=from_iso(r["created_at"]),
        run_id=r["run_id"],
        tags=tuple(t for t in r["tags"].split(",") if t),
        payload=json.loads(r["payload"]),
        consolidated=bool(r["consolidated"]),
    )


def _procedure(r: sqlite3.Row) -> Procedure:
    return Procedure(
        id=r["id"],
        scope=r["scope"],
        name=r["name"],
        description=r["description"],
        successes=r["successes"],
        failures=r["failures"],
        updated_at=from_iso(r["updated_at"]),
    )
