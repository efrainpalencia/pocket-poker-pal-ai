from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import (GENERATE, GRADE, RETRIEVE, RETRY_OR_CLARIFY,
                          ROUTE_OR_CLARIFY, TRY_OTHER_NAMESPACE)
from graph.nodes import generate, retrieve
from graph.nodes.grade import grade
from graph.nodes.retry_or_clarify import retry_or_clarify
from graph.nodes.route_or_clarify import route_or_clarify
from graph.nodes.try_other_namespace import try_other_namespace
from graph.state import GraphState

load_dotenv()

MAX_RETRIES = 2  # keep if used inside nodes/logic


def after_route(state: GraphState) -> str:
    return (
        "needs_user_clarification"
        if state.get("needs_clarification")
        else "proceed_to_retrieve"
    )


def after_retry(state: GraphState) -> str:
    return "end" if state.get("force_end") else "retrieve"


def after_grade(state: GraphState) -> str:
    if state.get("force_end"):
        return "max_retries_reached"

    gs = state.get("generation_structured")
    not_found = isinstance(gs, dict) and gs.get("mode") == "not_found"

    confidence = float(state.get("confidence", 0.0))
    grounded = bool(state.get("grounded", False))

    if confidence >= 0.60 and grounded:
        return "answer_accepted"

    # try opposite namespace once
    if (not_found or not grounded) and not state.get("fallback_attempted"):
        return "try_other_namespace"

    return "low_confidence_retry"


def build_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node(ROUTE_OR_CLARIFY, route_or_clarify)
    workflow.add_node(RETRIEVE, retrieve)
    workflow.add_node(GENERATE, generate)
    workflow.add_node(GRADE, grade)
    workflow.add_node(RETRY_OR_CLARIFY, retry_or_clarify)
    workflow.add_node(TRY_OTHER_NAMESPACE, try_other_namespace)

    workflow.set_entry_point(ROUTE_OR_CLARIFY)

    workflow.add_conditional_edges(
        ROUTE_OR_CLARIFY,
        after_route,
        {
            "proceed_to_retrieve": RETRIEVE,
            "needs_user_clarification": RETRY_OR_CLARIFY,
        },
    )

    workflow.add_edge(RETRIEVE, GENERATE)
    workflow.add_edge(GENERATE, GRADE)
    workflow.add_edge(TRY_OTHER_NAMESPACE, RETRIEVE)

    workflow.add_conditional_edges(
        GRADE,
        after_grade,
        {
            "answer_accepted": END,
            "try_other_namespace": TRY_OTHER_NAMESPACE,
            "low_confidence_retry": RETRY_OR_CLARIFY,
            "max_retries_reached": END,
        },
    )

    workflow.add_conditional_edges(
        RETRY_OR_CLARIFY,
        after_retry,
        {
            "retrieve": RETRIEVE,
            "end": END,
        },
    )

    return workflow


def build_graph(checkpointer=None):
    """
    Compile a runnable graph instance.
    Pass `checkpointer` from your lifespan (recommended).
    """
    workflow = build_workflow()
    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


def write_graph_png(
    output_file: str = "graph.png",
    checkpointer=None,
    *,
    xray: bool = False,
) -> str:
    """
    Build/compile the graph and write a Mermaid PNG diagram to disk.

    Call this from main.py whenever you want to update graph.png.

    Args:
        output_file: path to write PNG to (relative or absolute).
        checkpointer: optional saver passed to build_graph().
        xray: if True, includes more internal detail in the diagram.

    Returns:
        Absolute path of the written file (as a string).
    """
    g = build_graph(checkpointer=checkpointer)
    out_path = Path(output_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # LangGraph provides get_graph().draw_mermaid_png(...)
    g.get_graph(xray=xray).draw_mermaid_png(output_file_path=str(out_path))

    return str(out_path)
