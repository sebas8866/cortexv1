"""`cortex` command line.

cortex agents                       list registered agents
cortex run ads_optimizer            run one cycle now and print the briefing
cortex approvals                    list pending approvals
cortex approve <id> --by you        approve (or --reject) a pending action
cortex memory search "budget"       ranked recall across all memory kinds
cortex memory facts campaign:c_101  current beliefs about a subject
cortex serve                        start the control-plane API
cortex worker                       start scheduler + worker (the 24/7 loop)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import sys
from collections.abc import Sequence

from cortex.agents import available_agents
from cortex.config import Settings, load_settings
from cortex.orchestration.queue import InMemoryQueue, build_queue
from cortex.orchestration.worker import Scheduler, Worker
from cortex.runtime.cortex import Cortex


def _kv(pairs: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--input expects key=value, got {pair!r}")
        out[key] = value
    return out


async def _run(settings: Settings, args: argparse.Namespace) -> int:
    cortex = Cortex(settings)
    try:
        record = await cortex.run_agent(args.agent, _kv(args.input), scope=args.scope)
        print(record.summary or record.error)
        print(
            f"\nrun {record.id}: {record.status.value} "
            f"({' -> '.join(s['node'] for s in record.trace)})"
        )
        return 0 if record.status.value == "succeeded" else 1
    finally:
        cortex.close()


async def _decide(settings: Settings, args: argparse.Namespace, approve: bool) -> int:
    cortex = Cortex(settings)
    try:
        approval = await cortex.decide(
            args.approval_id, approve=approve, by=args.by, note=args.note
        )
        print(f"{approval.id}: {approval.status.value} by {approval.decided_by}")
        return 0
    finally:
        cortex.close()


async def _worker(settings: Settings) -> None:
    cortex = Cortex(settings)
    queue = build_queue(settings.queue)
    if isinstance(queue, InMemoryQueue) and not settings.schedules:
        logging.warning("in-memory queue with no schedules: this worker will idle")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Windows has no loop signal handlers; Ctrl+C arrives as KeyboardInterrupt instead.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    scheduler = Scheduler(settings.schedules, queue, default_scope=settings.default_scope)
    worker = Worker(cortex, queue)
    try:
        await asyncio.gather(scheduler.run_forever(stop), worker.run_forever(stop))
    finally:
        cortex.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cortex", description="Cortex agent operating system")
    p.add_argument("--config", help="path to cortex.yaml (default: $CORTEX_CONFIG or ./config)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("agents", help="list registered agents")

    run = sub.add_parser("run", help="run one agent cycle now")
    run.add_argument("agent")
    run.add_argument("--scope")
    run.add_argument("--input", action="append", default=[], metavar="KEY=VALUE")

    sub.add_parser("approvals", help="list pending approvals")
    for name in ("approve", "reject"):
        d = sub.add_parser(name, help=f"{name} a pending action")
        d.add_argument("approval_id")
        d.add_argument("--by", required=True)
        d.add_argument("--note", default="")

    mem = sub.add_parser("memory", help="inspect memory")
    msub = mem.add_subparsers(dest="mem_cmd", required=True)
    search = msub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--scope")
    search.add_argument("--limit", type=int, default=10)
    facts = msub.add_parser("facts")
    facts.add_argument("subject")
    facts.add_argument("--scope")
    stats = msub.add_parser("stats")
    stats.add_argument("--scope")

    serve = sub.add_parser("serve", help="start the control-plane API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("worker", help="run scheduler + worker until stopped")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings(args.config)

    if args.cmd == "agents":
        for name, desc in available_agents().items():
            print(f"{name:20} {desc}")
        return 0
    if args.cmd == "run":
        return asyncio.run(_run(settings, args))
    if args.cmd in ("approve", "reject"):
        return asyncio.run(_decide(settings, args, approve=args.cmd == "approve"))
    if args.cmd == "serve":
        import uvicorn

        from cortex.api import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return 0
    if args.cmd == "worker":
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(_worker(settings))
        return 0

    cortex = Cortex(settings)
    try:
        scope = getattr(args, "scope", None) or settings.default_scope
        if args.cmd == "approvals":
            for a in cortex.control.list_approvals():
                print(f"{a.id}  {a.action.kind:16} {a.action.target:24} {a.reason}")
        elif args.mem_cmd == "search":
            for h in cortex.memory.recall(args.query, scope, limit=args.limit):
                print(f"{h.score:.3f}  {h.kind.value:10} {h.text}")
        elif args.mem_cmd == "facts":
            for f in cortex.memory.facts_about(scope, args.subject):
                print(f"{f.predicate:28} {f.value:28} conf={f.confidence:.2f} n={f.evidence_count}")
        elif args.mem_cmd == "stats":
            print(json.dumps(cortex.memory.stats(scope), indent=2))
    finally:
        cortex.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
