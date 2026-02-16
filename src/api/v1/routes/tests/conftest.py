import json
import os
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import routers inside the app fixture to avoid heavy startup work at import time
chat_router = None
chat_stream_router = None

REQUIRED_ENV = ["OPENAI_API_KEY", "PINECONE_API_KEY", "INDEX_NAME"]


def missing_env() -> list[str]:
    return [k for k in REQUIRED_ENV if not os.getenv(k)]


_missing = missing_env()
skip_if_missing_env = pytest.mark.skipif(
    bool(_missing),
    reason=f"Missing env vars: {_missing}",
)


def _detect_base_prefix(app: FastAPI) -> str:
    """
    Detect the API base prefix by looking for the qa routes.
    Examples:
      /qa
      /api/v1/qa
      /api/v1/chat/qa
    Returns the prefix portion before '/qa'.
    """
    candidates: list[str] = []
    for r in app.routes:
        path = getattr(r, "path", "")
        if path.endswith("/qa") or path.endswith("/qa/stream"):
            candidates.append(path)

    candidates.sort(key=len)

    for p in candidates:
        if "/qa" in p:
            return p.split("/qa")[0]  # "" or "/api/v1" or "/api/v1/chat"
    return ""


@pytest.fixture(scope="session")
def app() -> FastAPI:
    """
    Minimal FastAPI app for API-level tests.
    Routers are included as-is; we auto-detect any base prefix.
    """
    # Ensure the graph checkpointer falls back to an in-memory saver during tests
    # to avoid DB connections or context-manager return values at import time.
    os.environ.setdefault("DATABASE_URL", "")

    # Import routers here so module-level side-effects (like building graph
    # singletons) happen after we control the environment.
    from api.v1.routes.chat import router as _chat_router
    from api.v1.routes.chat_stream import router as _chat_stream_router

    a = FastAPI()
    a.include_router(_chat_router)
    a.include_router(_chat_stream_router)
    return a


@pytest.fixture(scope="session")
def client(app: FastAPI):
    return TestClient(app)


@pytest.fixture(scope="session")
def base_prefix(app: FastAPI) -> str:
    return _detect_base_prefix(app)


def new_thread_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


# ----------------------------
# SSE helpers
# ----------------------------


def iter_sse_json(resp):
    """
    Yields decoded JSON objects from an SSE response.
    Expects lines like: 'data: { ...json... }'
    """
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if not payload:
            continue
        yield json.loads(payload)


def collect_until_terminal(resp, max_events: int = 25):
    """
    Collects SSE events until a terminal event type is found or max_events is reached.

    Returns:
      (events, terminal) where terminal is the terminal event dict or None.
    """
    events: list[dict] = []
    terminal: dict | None = None

    for evt in iter_sse_json(resp):
        events.append(evt)

        t = evt.get("type")
        if t in {"complete", "needs_clarification", "error"}:
            terminal = evt
            break

        if len(events) >= max_events:
            break

    return events, terminal


def collect_until_terminal_strict(resp, max_events: int = 25):
    """
    Same as collect_until_terminal but asserts a terminal event was reached.
    """
    events, terminal = collect_until_terminal(resp, max_events=max_events)
    assert terminal is not None, (
        f"Did not reach terminal event within {max_events} events. "
        f"Last event: {events[-1] if events else None}"
    )
    return events, terminal


# ----------------------------
# Contract helpers
# ----------------------------


def assert_json_contract(data: dict, thread_id: str):
    assert data.get("thread_id") == thread_id
    assert data.get("status") in {"complete", "needs_clarification"}

    if data["status"] == "complete":
        assert data.get("generation")
        assert data.get("grounded") is True
        assert float(data.get("confidence", 0.0)) >= 0.60
    else:
        prompt = data.get("prompt") or {}
        assert prompt.get("type") in {"choose_ruleset", "free_text"}


def assert_sse_terminal_contract(terminal: dict, thread_id: str):
    assert terminal is not None
    assert terminal.get("thread_id") == thread_id
    assert terminal.get("type") in {"complete", "needs_clarification", "error"}

    if terminal["type"] == "complete":
        # Required for complete
        assert terminal.get("generation")

        # Optional fields (validate shape only if present)
        if "grounded" in terminal:
            assert isinstance(terminal["grounded"], bool)
        if "confidence" in terminal:
            # allow int/float/str that can be cast
            float(terminal["confidence"])

    elif terminal["type"] == "needs_clarification":
        prompt = terminal.get("prompt") or {}
        assert prompt.get("type") in {"choose_ruleset", "free_text"}

    else:  # error
        assert terminal.get("message") or terminal.get("error")


def debug_routes(app: FastAPI) -> list[str]:
    """
    Useful for debugging: print this list in a failing test.
    """
    paths: list[str] = []
    for r in app.routes:
        p = getattr(r, "path", None)
        if p:
            paths.append(p)
    return sorted(paths)
