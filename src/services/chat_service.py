from typing import Any, Dict
from uuid import uuid4

from langgraph.types import Command

from graph.graph import graph


def cfg(thread_id: str) -> Dict[str, Any]:
    """Return a minimal langgraph configuration for a given thread.

    Args:
        thread_id: The thread identifier to scope graph state/configuration.

    Returns:
        A dict suitable for passing as `config` to `graph.invoke`.
    """

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
    """Invoke the graph with a question and return a QA-style result.

    The function will create a new `thread_id` if one is not provided,
    call the central `graph.invoke`, and convert interrupt outputs into
    a `needs_clarification` response when appropriate.

    Args:
        question: User question text.
        thread_id: Optional thread identifier to resume state.

    Returns:
        A dictionary conforming to `QAOut`-like shape with `status`,
        `thread_id`, and either `prompt` (if clarification required)
        or `generation`/metadata when complete.
    """

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
    """Resume an interrupted thread by sending a reply command to the graph.

    Args:
        thread_id: The thread identifier to resume.
        reply: The user's reply text used to continue processing.

    Returns:
        A dict similar to `ask_question`, indicating `needs_clarification`
        or `complete` and including generation/confidence metadata.
    """

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
