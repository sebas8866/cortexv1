"""Control-plane state: runs, approvals and the execution audit log."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from cortex.agents.base import ProposedAction
from cortex.db import Database, from_iso, new_id, to_iso, utcnow
from cortex.orchestration.executor import ExecutionResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    agent       TEXT NOT NULL,
    scope       TEXT NOT NULL,
    status      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    summary     TEXT NOT NULL DEFAULT '',
    trace       TEXT NOT NULL DEFAULT '[]',
    error       TEXT
);
CREATE INDEX IF NOT EXISTS runs_recent ON runs (started_at DESC);

CREATE TABLE IF NOT EXISTS approvals (
    id         TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES runs (id),
    scope      TEXT NOT NULL,
    action     TEXT NOT NULL,
    kind       TEXT NOT NULL,
    target     TEXT NOT NULL,
    reason     TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    decided_at TEXT,
    decided_by TEXT,
    note       TEXT
);
CREATE INDEX IF NOT EXISTS approvals_open ON approvals (status, scope);

CREATE TABLE IF NOT EXISTS executions (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    scope       TEXT NOT NULL,
    action_id   TEXT NOT NULL,
    kind        TEXT NOT NULL,
    target      TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    dry_run     INTEGER NOT NULL,
    detail      TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS executions_scope_time ON executions (scope, created_at);
"""


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(slots=True)
class RunRecord:
    id: str
    agent: str
    scope: str
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None
    summary: str
    trace: list[dict[str, Any]]
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "scope": self.scope,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "summary": self.summary,
            "trace": self.trace,
            "error": self.error,
        }


@dataclass(slots=True)
class Approval:
    id: str
    run_id: str
    scope: str
    action: ProposedAction
    reason: str
    status: ApprovalStatus
    created_at: datetime
    decided_at: datetime | None
    decided_by: str | None
    note: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "scope": self.scope,
            "action": self.action.to_dict(),
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "decided_by": self.decided_by,
            "note": self.note,
        }


class ControlStore:
    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] = utcnow) -> None:
        self._db = Database(path, _SCHEMA)
        self.clock = clock

    # ---- runs

    def create_run(self, agent: str, scope: str) -> RunRecord:
        run_id = new_id("run")
        with self._db.tx() as c:
            c.execute(
                "INSERT INTO runs (id, agent, scope, status, started_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, agent, scope, RunStatus.RUNNING.value, to_iso(self.clock())),
            )
        return self.get_run(run_id)

    def finish_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        summary: str,
        trace: Sequence[dict[str, Any]],
        error: str | None = None,
    ) -> RunRecord:
        with self._db.tx() as c:
            c.execute(
                "UPDATE runs SET status = ?, finished_at = ?, summary = ?, trace = ?, error = ?"
                " WHERE id = ?",
                (
                    status.value,
                    to_iso(self.clock()),
                    summary,
                    json.dumps(list(trace)),
                    error,
                    run_id,
                ),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        rows = self._db.query("SELECT * FROM runs WHERE id = ?", (run_id,))
        if not rows:
            raise KeyError(f"run {run_id} not found")
        return _run(rows[0])

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        return [
            _run(r)
            for r in self._db.query("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,))
        ]

    # ---- approvals

    def add_approval(
        self, run_id: str, scope: str, action: ProposedAction, reason: str
    ) -> Approval | None:
        """Queue an action for review. Returns None if the same kind+target is already pending."""
        with self._db.tx() as c:
            dup = c.execute(
                "SELECT 1 FROM approvals WHERE scope = ? AND kind = ? AND target = ?"
                " AND status = 'pending'",
                (scope, action.kind, action.target),
            ).fetchone()
            if dup:
                return None
            approval_id = new_id("apr")
            c.execute(
                "INSERT INTO approvals (id, run_id, scope, action, kind, target, reason,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    approval_id,
                    run_id,
                    scope,
                    json.dumps(action.to_dict()),
                    action.kind,
                    action.target,
                    reason,
                    to_iso(self.clock()),
                ),
            )
        return self.get_approval(approval_id)

    def get_approval(self, approval_id: str) -> Approval:
        rows = self._db.query("SELECT * FROM approvals WHERE id = ?", (approval_id,))
        if not rows:
            raise KeyError(f"approval {approval_id} not found")
        return _approval(rows[0])

    def list_approvals(
        self, status: ApprovalStatus | None = ApprovalStatus.PENDING
    ) -> list[Approval]:
        if status is None:
            rows = self._db.query("SELECT * FROM approvals ORDER BY created_at DESC")
        else:
            rows = self._db.query(
                "SELECT * FROM approvals WHERE status = ? ORDER BY created_at DESC", (status.value,)
            )
        return [_approval(r) for r in rows]

    def decide(self, approval_id: str, *, approve: bool, by: str, note: str = "") -> Approval:
        status = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
        with self._db.tx() as c:
            cur = c.execute(
                "UPDATE approvals SET status = ?, decided_at = ?, decided_by = ?, note = ?"
                " WHERE id = ? AND status = 'pending'",
                (status.value, to_iso(self.clock()), by, note, approval_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"approval {approval_id} is not pending")
        return self.get_approval(approval_id)

    # ---- audit

    def record_execution(
        self, run_id: str, scope: str, result: ExecutionResult, *, approved_by: str
    ) -> None:
        with self._db.tx() as c:
            c.execute(
                "INSERT INTO executions (id, run_id, scope, action_id, kind, target, ok, dry_run,"
                " detail, approved_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id("exe"),
                    run_id,
                    scope,
                    result.action_id,
                    result.kind,
                    result.target,
                    int(result.ok),
                    int(result.dry_run),
                    result.detail,
                    approved_by,
                    to_iso(self.clock()),
                ),
            )

    def count_auto_executions(self, scope: str, *, hours: int) -> int:
        since = to_iso(self.clock() - timedelta(hours=hours))
        rows = self._db.query(
            "SELECT COUNT(*) n FROM executions WHERE scope = ? AND approved_by = 'policy'"
            " AND created_at >= ?",
            (scope, since),
        )
        return int(rows[0]["n"])

    def list_executions(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._db.query("SELECT * FROM executions ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._db.close()


def _run(r: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=r["id"],
        agent=r["agent"],
        scope=r["scope"],
        status=RunStatus(r["status"]),
        started_at=from_iso(r["started_at"]),
        finished_at=from_iso(r["finished_at"]) if r["finished_at"] else None,
        summary=r["summary"],
        trace=json.loads(r["trace"]),
        error=r["error"],
    )


def _approval(r: sqlite3.Row) -> Approval:
    return Approval(
        id=r["id"],
        run_id=r["run_id"],
        scope=r["scope"],
        action=ProposedAction.from_dict(json.loads(r["action"])),
        reason=r["reason"],
        status=ApprovalStatus(r["status"]),
        created_at=from_iso(r["created_at"]),
        decided_at=from_iso(r["decided_at"]) if r["decided_at"] else None,
        decided_by=r["decided_by"],
        note=r["note"],
    )
