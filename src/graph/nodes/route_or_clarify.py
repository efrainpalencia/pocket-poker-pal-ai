from graph.chains.classifier import classifier_chain
from graph.consts import SEMINOLE_NAMESPACE, TDA_NAMESPACE
from graph.state import GraphState

TOURNAMENT_HINTS = [
    "tournament",
    "level",
    "levels",
    "late reg",
    "late registration",
    "registration",
    "bag",
    "bagging",
    "break",
    "icm",
    "payout",
    "ante",
    "button ante",
    "bb ante",
    "re-entry",
    "rebuy",
    "add-on",
    "chip race",
]

CASH_HINTS = [
    "cash game",
    "rake",
    "time rake",
    "must-move",
    "table stakes",
    "buy-in",
    "straddle",
    "missed blind",
    "seat change",
    "runner",
    "comp",
    "time charge",
]


def heuristic_classify(question: str) -> str:
    """Quick heuristic classification of a question into a game type.

    Looks for common tournament/cash hints in the lowercase question text.
    """

    q = (question or "").lower()
    if any(h in q for h in TOURNAMENT_HINTS):
        return "tournament"
    if any(h in q for h in CASH_HINTS):
        return "cash-game"
    return "unknown"


def normalize_game_type(raw: str) -> str:
    """Normalise raw classifier output into canonical game type.

    Returns one of: `tournament`, `cash-game` or `unknown`.
    """

    if not raw:
        return "unknown"

    s = raw.strip().lower()

    if s in {"tournament", "tourney"}:
        return "tournament"
    if s in {"cash-game", "cash game", "cash", "cashgame"}:
        return "cash-game"
    if s in {"unknown", "unsure", "unclear"}:
        return "unknown"

    if "tournament" in s:
        return "tournament"
    if "cash" in s:
        return "cash-game"

    return "unknown"


def route_or_clarify(state: GraphState) -> dict:
    """Graph node: decide routing to a ruleset or ask for clarification.

    Uses heuristics and an LLM classifier as a fallback to choose the
    `game_type` and `namespace` for retrieval. When unsure, returns a
    `needs_clarification` payload prompting the UI to ask the user.
    """

    # ✅ Short-circuit if UI (or prior step) already set explicit routing fields
    existing_game_type = state.get("game_type")
    existing_namespace = state.get("namespace")

    if existing_game_type in {"tournament", "cash-game"} and existing_namespace:
        # Ensure needs_clarification is off; keep meta_filter if present
        return {
            "needs_clarification": False,
            "game_type": existing_game_type,
            "namespace": existing_namespace,
            "meta_filter": state.get("meta_filter") or {},
        }

    q = state.get("question", "")

    game_type = heuristic_classify(q)

    if game_type == "unknown":
        raw = classifier_chain.invoke({"question": q})
        game_type = normalize_game_type(raw)

    if game_type == "tournament":
        return {
            "needs_clarification": False,
            "game_type": "tournament",
            "namespace": TDA_NAMESPACE,
            "meta_filter": {},
        }

    if game_type == "cash-game":
        return {
            "needs_clarification": False,
            "game_type": "cash-game",
            "namespace": SEMINOLE_NAMESPACE,
            "meta_filter": {"game_type": "cash-game"},
        }

    return {
        "needs_clarification": True,
        "missing_info": ["Is this about tournament rules or cash-game rules?"],
    }
