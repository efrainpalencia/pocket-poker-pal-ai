import json
from uuid import uuid4
from typing import Any, AsyncIterator

from langgraph.types import Command
from graph.graph import graph


DEFAULT_CLARIFY_PROMPT = (
    "Quick clarification: is this about **tournament rules** or **cash game rules**? "
    "Reply with 'tournament' or 'cash-game'."
)


def cfg(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _extract_token(event: dict) -> str | None:
    if event.get("event") != "on_chat_model_stream":
        return None
    chunk = event.get("data", {}).get("chunk")
    if not chunk:
        return None
    text = getattr(chunk, "content", None)
    return text if text else None


def _extract_interrupt(event: dict) -> str | None:
    if event.get("event") != "on_chain_end":
        return None
    output = event.get("data", {}).get("output")
    if isinstance(output, dict) and "__interrupt__" in output:
        intr = output["__interrupt__"][0]
        return str(intr.value)
    return None


def _state(thread_id: str):
    return graph.get_state(cfg(thread_id))


def _final_generation(thread_id: str) -> str | None:
    # ✅ generation is stored in state.values
    return _state(thread_id).values.get("generation")


def _is_interrupted(thread_id: str) -> tuple[bool, str | None]:
    st = _state(thread_id)

    # 1) Some LangGraph versions put interrupt info in tasks
    tasks = getattr(st, "tasks", None) or []
    for t in tasks:
        intrs = getattr(t, "interrupts", None) or []
        if intrs:
            return True, str(intrs[0].value)

    # 2) Heuristic: no generation + has next node => likely awaiting resume
    vals = getattr(st, "values", {}) or {}
    if vals.get("generation") is None and getattr(st, "next", None):
        return True, None

    return False, None


async def stream_qa(question: str, thread_id: str | None = None) -> AsyncIterator[str]:
    thread_id = thread_id or str(uuid4())
    yield sse({"type": "start", "thread_id": thread_id})

    async for event in graph.astream_events(
        {"question": question, "game_type": None},
        config=cfg(thread_id),
        version="v2",
    ):
        token = _extract_token(event)
        if token:
            yield sse({"type": "token", "thread_id": thread_id, "text": token})

        prompt = _extract_interrupt(event)
        if prompt:
            yield sse({
                "type": "needs_clarification",
                "thread_id": thread_id,
                "prompt": prompt,
            })
            return

    interrupted, prompt = _is_interrupted(thread_id)
    if interrupted:
        yield sse({
            "type": "needs_clarification",
            "thread_id": thread_id,
            "prompt": prompt or DEFAULT_CLARIFY_PROMPT,
        })
        return

    yield sse({
        "type": "complete",
        "thread_id": thread_id,
        "generation": _final_generation(thread_id),
    })


async def stream_resume(thread_id: str, reply: str) -> AsyncIterator[str]:
    yield sse({"type": "resume", "thread_id": thread_id})

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
            yield sse({
                "type": "needs_clarification",
                "thread_id": thread_id,
                "prompt": prompt,
            })
            return

    interrupted, prompt = _is_interrupted(thread_id)
    if interrupted:
        yield sse({
            "type": "needs_clarification",
            "thread_id": thread_id,
            "prompt": prompt or DEFAULT_CLARIFY_PROMPT,
        })
        return

    yield sse({
        "type": "complete",
        "thread_id": thread_id,
        "generation": _final_generation(thread_id),
    })
