from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from graph.state import GraphState
from graph.nodes import generate, retrieve
from graph.nodes.grade import grade
from graph.nodes.retry_or_clarify import retry_or_clarify
from graph.nodes.route_or_clarify import route_or_clarify
from graph.consts import GENERATE, GRADE, RETRIEVE, RETRY_OR_CLARIFY, ROUTE_OR_CLARIFY

load_dotenv()


def after_grade(state: GraphState) -> str:
    conf = state.get("confidence", 0.0)
    return END if conf >= 0.60 else "retry_or_clarify"


workflow = StateGraph(GraphState)
workflow.add_node(ROUTE_OR_CLARIFY, route_or_clarify)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GENERATE, generate)
workflow.add_node(GRADE, grade)
workflow.add_node(RETRY_OR_CLARIFY, retry_or_clarify)

workflow.set_entry_point(ROUTE_OR_CLARIFY)
workflow.add_edge(ROUTE_OR_CLARIFY, RETRIEVE)
workflow.add_edge(RETRIEVE, GENERATE)
workflow.add_edge(GENERATE, GRADE)
workflow.add_conditional_edges(GRADE, after_grade)
workflow.add_edge(RETRY_OR_CLARIFY, RETRIEVE)

graph = workflow.compile(checkpointer=MemorySaver())
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
