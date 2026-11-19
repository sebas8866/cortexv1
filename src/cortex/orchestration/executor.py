"""Action execution with an audit trail and memory write-back.

Handlers are registered per action kind. In dry-run mode (the default) every kind
succeeds without touching external systems, which is how the example flow runs.
Live mode requires a registered adapter for the kind; there is no silent fallback.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from cortex.agents.base import ProposedAction
from cortex.memory import MemoryStore

log = logging.getLogger(__name__)

Handler = Callable[[ProposedAction], Awaitable[str]]


@dataclass(slots=True, frozen=True)
class ExecutionResult:
    action_id: str
    kind: str
    target: str
    ok: bool
    dry_run: bool
    detail: str


class ActionExecutor:
    def __init__(self, memory: MemoryStore, *, dry_run: bool = True) -> None:
        self.memory = memory
        self.dry_run = dry_run
        self._handlers: dict[str, Handler] = {}

    def register(self, kind: str, handler: Handler) -> None:
        self._handlers[kind] = handler

    async def execute(self, action: ProposedAction, scope: str) -> ExecutionResult:
        try:
            if self.dry_run:
                detail = f"dry-run: would {action.kind} {action.target} {action.params}"
            elif handler := self._handlers.get(action.kind):
                detail = await handler(action)
            else:
                raise LookupError(f"no live adapter registered for '{action.kind}'")
            ok = True
        except Exception as exc:  # adapters are external code; record, never crash the run
            log.exception("action %s failed", action.id)
            ok, detail = False, f"{type(exc).__name__}: {exc}"

        if ok:
            # What Cortex did is known, not inferred: write effects as full-confidence facts.
            for effect in action.effects:
                self.memory.assert_fact(
                    scope, effect.subject, effect.predicate, effect.value, confidence=1.0
                )
        self.memory.record_procedure_outcome(
            scope, action.kind, success=ok, description=f"{action.agent} action"
        )
        return ExecutionResult(action.id, action.kind, action.target, ok, self.dry_run, detail)
