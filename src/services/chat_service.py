from typing import Any, Dict, Optional
from uuid import uuid4

from langgraph.types import Command

from api.core.rate_limit import _client_ip
from api.core.thread_token import create_thread_token, verify_thread_token


def cfg(thread_id: str) -> Dict[str, Any]:
    """Return a minimal langgraph configuration for a given thread."""
    return {"configurable": {"thread_id": thread_id}}


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


def _extract_interrupt_from_output(out: dict) -> dict | None:
    """Detect LangGraph interrupt payload from invoke() result."""
    if isinstance(out, dict) and "__interrupt__" in out and out["__interrupt__"]:
        intr = out["__interrupt__"][0]
        value = getattr(intr, "value", None)

        if isinstance(value, dict):
            return value

        if isinstance(value, str) and value.strip():
            return {"type": "free_text", "message": value.strip()}

    return None


def ask_question(question: str, request=None, thread_id: Optional[str] = None) -> dict:
    """Invoke the graph with a question and return a QA-style result."""
    graph = _get_graph(request)

    thread_id = thread_id or str(uuid4())

    ip = _client_ip(request) if request else None
    thread_token = create_thread_token(thread_id=thread_id, ip=ip)

    out = graph.invoke(
        {"question": question},
        config=cfg(thread_id),
    )

    interrupt = _extract_interrupt_from_output(out)
    if interrupt:
        return {
            "status": "needs_clarification",
            "thread_id": thread_id,
            "thread_token": thread_token,
            "prompt": interrupt,
        }

    return {
        "status": "complete",
        "thread_id": thread_id,
        "thread_token": thread_token,
        "generation": out.get("generation"),
        "confidence": out.get("confidence"),
        "grounded": out.get("grounded"),
    }


def resume_question(
    thread_id: str, thread_token: str, reply: str, request=None
) -> dict:
    """Resume an interrupted thread by sending a reply command to the graph."""
    graph = _get_graph(request)

    ip = _client_ip(request) if request else None
    verify_thread_token(token=thread_token, thread_id=thread_id, ip=ip)

    out = graph.invoke(
        Command(resume=reply),
        config=cfg(thread_id),
    )

    interrupt = _extract_interrupt_from_output(out)
    if interrupt:
        return {
            "status": "needs_clarification",
            "thread_id": thread_id,
            "thread_token": thread_token,
            "prompt": interrupt,
        }

    return {
        "status": "complete",
        "thread_id": thread_id,
        "thread_token": thread_token,
        "generation": out.get("generation"),
        "confidence": out.get("confidence"),
        "grounded": out.get("grounded"),
    }
