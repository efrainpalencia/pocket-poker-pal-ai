from uuid import uuid4
from typing import Any

from langgraph.types import Command
from graph.graph import graph
from schemas.chat_schema import AskIn, ResumeIn, QAOut


def cfg(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def to_qa_out(result: dict[str, Any], thread_id: str) -> QAOut:
    if "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        return QAOut(
            status="needs_clarification",
            thread_id=thread_id,
            prompt=str(intr.value),
        )

    return QAOut(
        status="complete",
        thread_id=thread_id,
        generation=result.get("generation"),
    )


def get_clarification(req: AskIn) -> QAOut:
    thread_id = req.thread_id or str(uuid4())

    result = graph.invoke(
        {"question": req.question, "game_type": None},
        config=cfg(thread_id),
    )

    return to_qa_out(result, thread_id)


def get_answer(req: ResumeIn) -> QAOut:
    result = graph.invoke(
        Command(resume=req.reply),
        config=cfg(req.thread_id),
    )

    return to_qa_out(result, req.thread_id)
