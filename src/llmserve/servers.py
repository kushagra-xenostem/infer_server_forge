"""Run ASGI apps with uvicorn in background threads (tests and the local simulator)."""

import time
import socket
import asyncio
import uvicorn
import threading
from typing import Any
from collections.abc import Iterator
from contextlib import contextmanager


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def serve(
    app: Any, startup_timeout_s: float = 10.0, port: int | None = None, log_level: str = "warning"
) -> Iterator[str]:
    """Serve `app` on localhost (a free port unless `port` is given); yield its base URL; stop it on exit."""
    port = port or free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level=log_level, lifespan="on"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    deadline = time.monotonic() + startup_timeout_s
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("server did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
