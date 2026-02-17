from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from langgraph.types import Command

from api.core.langgraph_runtime import ensure_graph, rebuild_graph
from api.core.rate_limit import _client_ip
from api.core.thread_token import create_thread_token, verify_thread_token


def cfg(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _extract_interrupt_from_output(out: dict) -> dict | None:
    if isinstance(out, dict) and "__interrupt__" in out:
        intr = out["__interrupt__"][0]
        value = getattr(intr, "value", None)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            return {"type": "free_text", "message": value.strip()}
    return None


def _looks_like_disconnect(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = [
        "the connection is closed",
        "ssl connection has been closed unexpectedly",
        "terminating connection due to administrator command",
        "server closed the connection unexpectedly",
        "connection reset by peer",
        "broken pipe",
        "connection not open",
        "consuming input failed",
    ]
    return any(n in msg for n in needles)


def _invoke_with_reconnect(request, invoke_fn):
    """
    Invoke once. If it fails due to DB disconnect, rebuild graph and retry once.
    """
    graph = ensure_graph(request)
    try:
        return invoke_fn(graph)
    except Exception as e:
        if not request or not _looks_like_disconnect(e):
            raise
        graph2 = rebuild_graph(request)
        return invoke_fn(graph2)


def ask_question(question: str, request=None, thread_id: str | None = None) -> dict:
    thread_id = thread_id or str(uuid4())

    ip = _client_ip(request) if request else None
    thread_token = create_thread_token(thread_id=thread_id, ip=ip)

    def _do(g):
        return g.invoke({"question": question}, config=cfg(thread_id))

    out = _invoke_with_reconnect(request, _do)

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
    ip = _client_ip(request) if request else None
    verify_thread_token(token=thread_token, thread_id=thread_id, ip=ip)

    def _do(g):
        return g.invoke(Command(resume=reply), config=cfg(thread_id))

    out = _invoke_with_reconnect(request, _do)

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
