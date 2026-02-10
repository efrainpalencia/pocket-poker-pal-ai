from typing import Any, Dict
from graph.state import GraphState
from graph.chains.grader import grader_chain


def grade(state: GraphState) -> dict:
    verdict = grader_chain.invoke({
        "question": state["question"],
        "context": "\n\n".join(state.get("documents", [])),
        "answer": state.get("generation", ""),
    }).strip().upper()

    if verdict.startswith("Y"):
        return {"grounded": True, "grade": "YES"}
    if verdict.startswith("P"):
        # treat PARTIAL as acceptable, but you may want to revise answer with a caveat
        return {"grounded": True, "grade": "PARTIAL"}
    return {"grounded": False, "grade": "NO"}
