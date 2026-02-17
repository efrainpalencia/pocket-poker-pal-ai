import json
from typing import Any, AsyncIterator, Optional, Tuple
from uuid import uuid4

from langgraph.types import Command

from api.core.rate_limit import _client_ip
from api.core.thread_token import create_thread_token, verify_thread_token

DEFAULT_CLARIFY_PROMPT = {
    "type": "choose_ruleset",
    "message": "I need to know which ruleset to use for your question.",
    "options": ["tournament", "cash-game"],
}


def cfg(thread_id: str) -> dict[str, Any]:
    """Return a minimal langgraph configuration for a given thread."""
    return {"configurable": {"thread_id": thread_id}}


def sse(data: dict) -> str:
    """Serialize a single SSE (Server-Sent Event) data payload."""
    return f"data: {json.dumps(data)}\n\n"


def _get_graph(request):
    """
    Retrieve the compiled LangGraph instance from FastAPI app state.

    Requires that main.py sets: app.state.graph = build_graph(...)
    """
    if request is None:
        raise RuntimeError("Request is required to access app.state.graph")

    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise RuntimeError(
            "LangGraph not initialized. Ensure main.py lifespan sets app.state.graph"
        )
    return graph


def _extract_token(event: dict) -> Optional[str]:
    """Extract a streamed token (text chunk) from a langgraph event."""
    if event.get("event") != "on_chat_model_stream":
        return None
    chunk = event.get("data", {}).get("chunk")
    if not chunk:
        return None
    text = getattr(chunk, "content", None)
    return text if text else None


def _extract_interrupt_from_output(output: Any) -> Optional[dict]:
    """
    LangGraph interrupt output:
      {"__interrupt__": [Interrupt(value=..., id=...)]}

    value may be a dict (structured) or a string.
    """
    if not (isinstance(output, dict) and "__interrupt__" in output and output["__interrupt__"]):
        return None

    intr = output["__interrupt__"][0]
    value = getattr(intr, "value", None)

    if isinstance(value, dict):
        return value

    if isinstance(value, str) and value.strip():
        return {"type": "free_text", "message": value.strip()}

    return None


def _extract_interrupt(event: dict) -> Optional[dict]:
    """Extract interrupt prompt from astream_events on_chain_end output."""
    if event.get("event") != "on_chain_end":
        return None
    output = event.get("data", {}).get("output")
    return _extract_interrupt_from_output(output)


def _state(graph, thread_id: str):
    """Return the current persisted graph state for a given thread."""
    return graph.get_state(cfg(thread_id))


def _final_generation(graph, thread_id: str) -> Optional[str]:
    """Return the final `generation` value from persisted state, if any."""
    return (_state(graph, thread_id).values or {}).get("generation")


def _is_interrupted(graph, thread_id: str) -> Tuple[bool, Optional[dict]]:
    """Determine whether a thread is currently interrupted."""
    st = _state(graph, thread_id)

    tasks = getattr(st, "tasks", None) or []
    for t in tasks:
        intrs = getattr(t, "interrupts", None) or []
        if intrs:
            value = getattr(intrs[0], "value", None)
            if isinstance(value, dict):
                return True, value
            if isinstance(value, str) and value.strip():
                return True, {"type": "free_text", "message": value.strip()}
            return True, None

    vals = getattr(st, "values", {}) or {}
    if vals.get("generation") is None and getattr(st, "next", None):
        return True, None

    return False, None


async def stream_qa(
    question: str, request, thread_id: str | None = None
) -> AsyncIterator[str]:
    """Asynchronously stream Server-Sent Events for a QA request."""
    graph = _get_graph(request)

    thread_id = thread_id or str(uuid4())

    ip = _client_ip(request)
    thread_token = create_thread_token(thread_id=thread_id, ip=ip)

    yield sse({"type": "start", "thread_id": thread_id, "thread_token": thread_token})

    async for event in graph.astream_events(
        {"question": question},
        config=cfg(thread_id),
        version="v2",
    ):
        token = _extract_token(event)
        if token:
            yield sse({"type": "token", "thread_id": thread_id, "text": token})

        prompt = _extract_interrupt(event)
        if prompt:
            yield sse(
                {
                    "type": "needs_clarification",
                    "thread_id": thread_id,
                    "thread_token": thread_token,
                    "prompt": prompt,
                }
            )
            return

    interrupted, prompt = _is_interrupted(graph, thread_id)
    if interrupted:
        yield sse(
            {
                "type": "needs_clarification",
                "thread_id": thread_id,
                "thread_token": thread_token,
                "prompt": prompt or DEFAULT_CLARIFY_PROMPT,
            }
        )
        return

    yield sse(
        {
            "type": "complete",
            "thread_id": thread_id,
            "thread_token": thread_token,
            "generation": _final_generation(graph, thread_id) or "",
        }
    )


async def stream_resume(
    thread_id: str, thread_token: str, reply: str, request
) -> AsyncIterator[str]:
    """Asynchronously stream Server-Sent Events when resuming a thread."""
    graph = _get_graph(request)

    ip = _client_ip(request)
    verify_thread_token(token=thread_token, thread_id=thread_id, ip=ip)

    yield sse({"type": "resume", "thread_id": thread_id, "thread_token": thread_token})

    async for event in graph.astream_events(
        Command(resume=reply),
        config=cfg(thread_id),
        version="v2",
    ):
        token = _extract_token(event)
        if token:
            yield sse({"type": "token", "thread_id": thread_id, "text": token})

        prompt = _extract_interrupt(event)
        if prompt:
            yield sse(
                {
                    "type": "needs_clarification",
                    "thread_id": thread_id,
                    "thread_token": thread_token,  # ✅ fixed typo
                    "prompt": prompt,
                }
            )
            return

    interrupted, prompt = _is_interrupted(graph, thread_id)
    if interrupted:
        yield sse(
            {
                "type": "needs_clarification",
                "thread_id": thread_id,
                "thread_token": thread_token,
                "prompt": prompt or DEFAULT_CLARIFY_PROMPT,
            }
        )
        return

    yield sse(
        {
            "type": "complete",
            "thread_id": thread_id,
            "thread_token": thread_token,
            "generation": _final_generation(graph, thread_id) or "",
        }
    )
