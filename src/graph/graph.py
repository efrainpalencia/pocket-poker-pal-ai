from dotenv import load_dotenv
from langgraph.types import interrupt
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from graph.state import GraphState
from graph.nodes import generate, retrieve
from graph.consts import GENERATE, RETRIEVE, ROUTE_TO_CLARIFY

load_dotenv()


def route_or_clarify(state: GraphState) -> dict:
    question = state["question"]

    # (A) your own heuristic / LLM classifier can go here
    game_type = state.get("game_type")
    if game_type not in ("tournament", "cash-game"):
        user_choice = interrupt(
            "Quick clarification: is this about **tournament rules** or **cash game rules**? "
            "Reply with 'tournament' or 'cash-game'."
        )

        # Validate + normalize (you can loop until valid if you want)
        choice = str(user_choice).strip().lower()
        if choice in ("tournament", "tourney", "t"):
            game_type = "tournament"
        elif choice in ("cash", "cashgame", "cash-game", "c"):
            game_type = "cash-game"
        else:
            # re-ask by interrupting again (loop pattern supported)
            # (keep it simple/serializable)
            user_choice = interrupt(
                f"'{user_choice}' wasn’t recognized. Please reply with 'tournament' or 'cash-game'."
            )
            choice = str(user_choice).strip().lower()
            game_type = "tournament" if "tourn" in choice else "cash-game"

    return {"game_type": game_type}


graph = StateGraph(GraphState)
graph.add_node(ROUTE_TO_CLARIFY, route_or_clarify)
graph.add_node(RETRIEVE, retrieve)
graph.add_node(GENERATE, generate)
graph.set_entry_point(ROUTE_TO_CLARIFY)
graph.add_edge(ROUTE_TO_CLARIFY, RETRIEVE)
graph.add_edge(RETRIEVE, GENERATE)
graph.add_edge(GENERATE, END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

print("Finished")
