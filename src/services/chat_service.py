from uuid import uuid4
from typing import Any, Dict

from langgraph.types import Command
from graph.graph import graph


def cfg(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _extract_interrupt_from_output(out: dict) -> dict | None:
    """
    Detect LangGraph interrupt payload from invoke() result.
    """
    if isinstance(out, dict) and "__interrupt__" in out:
        intr = out["__interrupt__"][0]
        value = getattr(intr, "value", None)

        if isinstance(value, dict):
            return value

        if isinstance(value, str) and value.strip():
            return {"type": "free_text", "message": value.strip()}

    return None


def ask_question(question: str, thread_id: str | None = None) -> dict:
    thread_id = thread_id or str(uuid4())

    out = graph.invoke(
        {"question": question},
        config=cfg(thread_id),
    )

    interrupt = _extract_interrupt_from_output(out)
    if interrupt:
        return {
            "status": "needs_clarification",
            "thread_id": thread_id,
            "prompt": interrupt,
        }

    return {
        "status": "complete",
        "thread_id": thread_id,
        "generation": out.get("generation"),
        "confidence": out.get("confidence"),
        "grounded": out.get("grounded"),
    }


def resume_question(thread_id: str, reply: str) -> dict:
    out = graph.invoke(
        Command(resume=reply),
        config=cfg(thread_id),
    )

    interrupt = _extract_interrupt_from_output(out)
    if interrupt:
        return {
            "status": "needs_clarification",
            "thread_id": thread_id,
            "prompt": interrupt,
        }

    return {
        "status": "complete",
        "thread_id": thread_id,
        "generation": out.get("generation"),
        "confidence": out.get("confidence"),
        "grounded": out.get("grounded"),
    }
