from graph.consts import SEMINOLE_NAMESPACE, TDA_NAMESPACE
from graph.state import GraphState


def try_other_namespace(state: GraphState) -> dict:
    current = state.get("game_type")

    if current == "cash-game":
        return {
            "fallback_attempted": True,
            "game_type": "tournament",
            "namespace": TDA_NAMESPACE,
            "meta_filter": {},
        }

    if current == "tournament":
        return {
            "fallback_attempted": True,
            "game_type": "cash-game",
            "namespace": SEMINOLE_NAMESPACE,
            "meta_filter": {"game_type": "cash-game"},
        }

    # If we don't even know, do nothing; retry_or_clarify will ask
    return {"fallback_attempted": True}
