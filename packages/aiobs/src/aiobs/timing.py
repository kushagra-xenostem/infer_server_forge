"""Timed spans that land in an EventLog, including when the block raises."""

import time
from typing import Any
from collections.abc import Iterator
from contextlib import contextmanager
from aiobs.events import Event, EventLog


@contextmanager
def span(log: EventLog | None, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
    """Time a block. Yields a dict the block may add attributes to; the event is written on exit,
    with status "error" and the exception type if the block raised (the exception still propagates)."""
    extra: dict[str, Any] = {}
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield extra
    except BaseException as e:
        status = "error"
        extra["error"] = type(e).__name__
        raise
    finally:
        if log is not None:
            log.write(
                Event(
                    kind="span", name=name, duration_s=time.perf_counter() - t0, status=status, attrs={**attrs, **extra}
                )
            )
