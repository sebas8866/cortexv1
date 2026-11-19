from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cortex.config import Settings, load_settings
from cortex.memory import MemoryStore
from cortex.runtime.cortex import Cortex

ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 18, 7, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def memory(tmp_path: Path, clock: FakeClock) -> Iterator[MemoryStore]:
    store = MemoryStore(tmp_path / "mem.db", clock=clock)
    yield store
    store.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = load_settings(ROOT / "config" / "cortex.yaml")
    s.data_dir = tmp_path
    s.llm.provider = "offline"
    return s


@pytest.fixture
def cortex(settings: Settings, clock: FakeClock) -> Iterator[Cortex]:
    cx = Cortex(settings, clock=clock)
    yield cx
    cx.close()


@pytest.fixture
def campaigns_csv() -> str:
    return str(ROOT / "examples" / "data" / "campaigns.csv")


@pytest.fixture
def auth_events() -> str:
    return str(ROOT / "examples" / "data" / "auth_events.jsonl")
