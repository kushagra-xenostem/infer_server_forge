"""The common event schema: one JSON object per line."""

import json
import time
import threading
from typing import Any
from pathlib import Path
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = 1


@dataclass
class Event:
    kind: str  # e.g. "span", "llm_call", "fault", "metric"
    name: str
    duration_s: float | None = None
    status: str = "ok"  # ok | error
    attrs: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)  # wall clock, seconds since the epoch
    schema: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, line: str) -> "Event":
        return cls(**json.loads(line))


class EventLog:
    """Append-only JSONL file; safe to share between threads of one process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: Event) -> None:
        line = event.to_json() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)

    def emit(self, kind: str, name: str, **attrs: Any) -> Event:
        ev = Event(kind=kind, name=name, attrs=attrs)
        self.write(ev)
        return ev

    def read(self) -> list[Event]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [Event.from_json(line) for line in lines if line.strip()]
