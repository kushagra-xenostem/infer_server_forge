import pytest
from pathlib import Path
from llmserve.servers import serve
from collections.abc import Iterator
from contextlib import contextmanager
from llmserve.gateway import Settings, create_app
from llmserve.mock_upstream import MockState, create_mock

FAST = 0.001  # mock time scale for tests: a 10 s simulated request takes 10 ms


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_state() -> MockState:
    return MockState(time_scale=FAST)


@pytest.fixture
def mock_url(mock_state: MockState) -> Iterator[str]:
    with serve(create_mock(mock_state)) as url:
        yield url


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "gateway.jsonl"


@pytest.fixture
def gateway_factory(mock_url: str, log_path: Path):  # type: ignore[no-untyped-def]
    @contextmanager
    def make(**overrides: object) -> Iterator[str]:
        upstream = str(overrides.pop("upstream", mock_url))
        settings = Settings(upstream=upstream, log_path=str(log_path), **overrides)  # type: ignore[arg-type]
        with serve(create_app(settings)) as url:
            yield url

    return make
