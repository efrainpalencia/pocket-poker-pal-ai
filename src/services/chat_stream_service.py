import json
from typing import Any, AsyncIterator, Optional, Tuple
from uuid import uuid4

from langgraph.types import Command

from api.core.rate_limit import _client_ip
from api.core.thread_token import create_thread_token, verify_thread_token
from graph.graph import graph

DEFAULT_CLARIFY_PROMPT = {
    "type": "choose_ruleset",
    "message": "I need to know which ruleset to use for your question.",
    "options": ["tournament", "cash-game"],
}


def cfg(thread_id: str) -> dict[str, Any]:
    """Return a minimal langgraph configuration for a given thread.

    Args:
        thread_id: The thread identifier to scope graph state/configuration.

    Returns:
        A dict suitable for passing as `config` to `graph.astream_events`.
    """

    return {"configurable": {"thread_id": thread_id}}


def sse(data: dict) -> str:
    """Serialize a single SSE (Server-Sent Event) data payload.

    Args:
        data: JSON-serializable mapping to send as the SSE `data` field.

    Returns:
        A string formatted as an SSE `data: ...\n\n` block.
    """

    return f"data: {json.dumps(data)}\n\n"


def _extract_token(event: dict) -> Optional[str]:
    """Extract a streamed token (text chunk) from a langgraph event.

    Returns the token string when present, otherwise `None`.
    """

    if event.get("event") != "on_chat_model_stream":
        return None
    chunk = event.get("data", {}).get("chunk")
    if not chunk:
        return None
    text = getattr(chunk, "content", None)
    return text if text else None


def _extract_interrupt(event: dict) -> Optional[dict]:
    """
    LangGraph interrupt comes back as:
      output = {"__interrupt__": [Interrupt(value=..., id=...)]}
    value may be a dict (structured) or a string.
    """
    if event.get("event") != "on_chain_end":
        return None

    output = event.get("data", {}).get("output")
    if not (isinstance(output, dict) and "__interrupt__" in output):
        return None

    intr = output["__interrupt__"][0]
    value = getattr(intr, "value", None)

    # If you used interrupt({...}) then value is already a dict
    if isinstance(value, dict):
        return value

    # Otherwise treat it as free_text prompt
    if isinstance(value, str) and value.strip():
        return {"type": "free_text", "message": value.strip()}

    return None


def _state(thread_id: str):
    """Return the current persisted graph state for a given thread.

    Args:
        thread_id: Thread identifier used to scope stored state.

    Returns:
        The LangGraph state object for the thread.
    """

    return graph.get_state(cfg(thread_id))


def _final_generation(thread_id: str) -> Optional[str]:
    """Return the final `generation` value from persisted state, if any."""

    return (_state(thread_id).values or {}).get("generation")


def _is_interrupted(thread_id: str) -> Tuple[bool, Optional[dict]]:
    """Determine whether a thread is currently interrupted.

    Returns a tuple `(is_interrupted, prompt)` where `prompt` is the
    interrupt payload (when available) or `None`.
    """

    st = _state(thread_id)

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
    question: str, request=None, thread_id: str | None = None
) -> AsyncIterator[str]:
    """Asynchronously stream Server-Sent Events for a QA request.

    Yields SSE-formatted strings representing `start`, `token`,
    `needs_clarification`, and `complete` events.
    """

    thread_id = thread_id or str(uuid4())

    ip = None
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

    interrupted, prompt = _is_interrupted(thread_id)
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
            "generation": _final_generation(thread_id) or "",
        }
    )


async def stream_resume(
    thread_id: str, thread_token: str, reply: str, request=None
) -> AsyncIterator[str]:
    """Asynchronously stream Server-Sent Events when resuming a thread.

    Yields SSE-formatted strings representing `resume`, incremental
    `token` events and final `complete` or `needs_clarification`.
    """
    ip = _client_ip(request) if request else None

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
                    "thead_token": thread_token,
                    "prompt": prompt,
                }
            )
            return

    interrupted, prompt = _is_interrupted(thread_id)
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
            "generation": _final_generation(thread_id) or "",
        }
    )
