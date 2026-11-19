# Architecture

## The cycle

Every agent run is the same six-node state graph (`cortex/orchestration/cycle.py`):

| Node | What happens | Writes |
|---|---|---|
| `decide` | The agent reads its inputs and memory, returns proposals, observations, notes and metrics. | nothing |
| `govern` | `ActionPolicy` gives each proposal `auto`, `review` or `deny`. | nothing |
| `act` | Auto-approved actions go to the executor. Skipped when there are none. | audit log, effect facts, procedure stats |
| `escalate` | `review` actions become pending approvals (deduplicated per kind + target). | approvals |
| `remember` | One episode for the run, carrying the observations; then consolidation. | episode, promoted facts |
| `report` | The LLM turns the run record into an operator briefing. | run summary |

Agents are deliberately side-effect free. That gives one place for guardrails, one
audit trail and one memory write path, and makes agents unit-testable against fixtures.

## Why a built-in graph runtime

The cycle needs routing, step limits, per-step timing and a trace that survives
failure. That is ~150 lines (`orchestration/graph.py`) and has no dependencies. Its API
mirrors LangGraph's `StateGraph` on purpose: when a workflow needs checkpointed
interrupts mid-cycle (pause for a human, resume next day), a LangGraph adapter can
compile the same node functions. That adapter is on the roadmap, not in the code.

## Autonomy and guardrails

Policy is evaluated per action, in this order:

1. `denied_kinds` → `deny`
2. risk above `max_auto_risk` → `review`
3. `change_pct` beyond `max_budget_change_pct` → `review`
4. auto-executed actions in the last 24h for this scope ≥ `daily_auto_action_limit` → `review`
5. otherwise `auto`

Agents assign risk; policy decides. A human decision on a review item is executed by
`Cortex.decide`, logged with the reviewer's name, and stored as a high-importance
episode so agents can recall it.

## Execution

`ActionExecutor` runs in dry-run mode by default. In live mode, an action kind with no
registered adapter fails loudly instead of falling back. On success, the action's
declared `effects` are written as facts with confidence 1.0, and the procedure track
record for that action kind is updated.

## Runtime topology

```
           ┌────────────┐      ┌──────────┐
 schedule ─►  Scheduler  ├─────►          │
           └────────────┘      │  Queue   ├──► Worker(s) ──► Cortex.run_agent
 POST /tasks ──────────────────►  (Redis) │
                               └──────────┘
 POST /runs ─────────────────────────────────────────────► Cortex.run_agent (inline)
```

Single node: `cortex worker` runs scheduler and worker in one process with the in-memory
queue. Multi-process: API, scheduler+worker and dashboard as separate containers on Redis
(see `docker-compose.yml`). Run exactly one scheduler.

## Data

One SQLite file (`var/cortex.db`, WAL mode) holds two logical stores:

- **Memory** (`memory/store.py`): `episodes`, `facts`, `fact_sources`, `procedures`,
  `memory_fts`.
- **Control** (`runtime/control.py`): `runs`, `approvals`, `executions`.

They are separate classes with separate connections so they can move to separate
databases (or Postgres schemas) independently.
