# Deployment

## Single machine (simplest)

```bash
pip install -e ".[llm]"
cortex serve --host 127.0.0.1 --port 8000 &
cortex worker
```

State lives in `data_dir` (default `./var`). Back up `var/cortex.db` (copy it while no
writer is active, or use `sqlite3 var/cortex.db ".backup backup.db"`).

## Docker compose

```bash
cp .env.example .env        # set ANTHROPIC_API_KEY if you want Claude-written briefings
docker compose up --build -d
```

| Service | Port | Notes |
|---|---|---|
| `api` | 8000 | Control plane. OpenAPI docs at `/docs`. |
| `worker` | - | Scheduler + worker on the Redis queue. Run one replica. |
| `dashboard` | 8501 | Streamlit, talks to `api` only. |
| `redis` | 6379 | Task queue, append-only persistence. |

`api` and `worker` share the `cortex-data` volume because both read and write the SQLite
database. That is the main reason to move to Postgres before scaling out.

## Configuration

`config/cortex.yaml` is the source of truth; environment variables override a few keys:

| Variable | Effect |
|---|---|
| `CORTEX_CONFIG` | Path to the YAML file. |
| `CORTEX_DATA_DIR` | Where the database lives. |
| `CORTEX_LLM_PROVIDER` | `offline` or `anthropic`. |
| `CORTEX_REDIS_URL` | Redis connection string. |
| `CORTEX_QUEUE_BACKEND` | `memory` or `redis`. |
| `ANTHROPIC_API_KEY` | Required when the provider is `anthropic`. Never put it in YAML. |

## Going live checklist

Cortex ships with `policy.dry_run: true`. Before turning it off:

1. Register a live adapter for every action kind your agents propose
   (`cortex.executor.register(kind, handler)`). Kinds without one fail rather than run.
2. Start with `max_auto_risk: low`, a small `daily_auto_action_limit` and a tight
   `max_budget_change_pct`. Widen them as the audit log builds a track record.
3. Put the API behind authentication. It has none of its own yet.
4. Run exactly one scheduler process.
5. Monitor `/runs` for `failed` status and `/approvals` for items waiting too long.

## Production gaps

See "Current limitations" in the README. The ones that matter for deployment: no API
auth, SQLite single-writer, in-process scheduler state, no live adapters.
