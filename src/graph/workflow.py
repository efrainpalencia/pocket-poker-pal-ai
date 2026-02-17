from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import GENERATE, GRADE, RETRIEVE, RETRY_OR_CLARIFY, ROUTE_OR_CLARIFY
from graph.nodes import generate, retrieve
from graph.nodes.grade import grade
from graph.nodes.retry_or_clarify import retry_or_clarify
from graph.nodes.route_or_clarify import route_or_clarify
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
    if isinstance(gs, dict) and gs.get("mode") == "not_found":
        return "insufficient_context"

    confidence = float(state.get("confidence", 0.0))
    grounded = bool(state.get("grounded", False))

    if confidence >= 0.60 and grounded:
        return "answer_accepted"

    return "low_confidence_retry"


def build_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node(ROUTE_OR_CLARIFY, route_or_clarify)
    workflow.add_node(RETRIEVE, retrieve)
    workflow.add_node(GENERATE, generate)
    workflow.add_node(GRADE, grade)
    workflow.add_node(RETRY_OR_CLARIFY, retry_or_clarify)

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

    workflow.add_conditional_edges(
        GRADE,
        after_grade,
        {
            "answer_accepted": END,
            "low_confidence_retry": RETRY_OR_CLARIFY,
            "insufficient_context": RETRY_OR_CLARIFY,
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
