from langgraph.types import interrupt
from graph.state import GraphState
from graph.chains.classifier import classifier_chain

TOURNAMENT_HINTS = (
    "tournament", "tda", "level", "levels", "blind level", "ante", "antes",
    "bag", "bagging", "late reg", "registration", "payout", "prize", "icm"
)
CASH_HINTS = (
    "cash", "rake", "time rake", "table stakes", "buy-in", "must move",
    "straddle", "seat change"
)


def _heuristic(question: str) -> str | None:
    q = question.lower()
    has_t = any(w in q for w in TOURNAMENT_HINTS)
    has_c = any(w in q for w in CASH_HINTS)
    if has_t and not has_c:
        return "tournament"
    if has_c and not has_t:
        return "cash-game"
    return None


def _normalize(choice: str) -> str | None:
    c = choice.strip().lower()
    if c in ("tournament", "tourney", "tda", "t"):
        return "tournament"
    if c in ("cash-game", "cashgame", "cash", "c"):
        return "cash-game"
    return None


def route_or_clarify(state: GraphState) -> dict:
    q = state["question"]
    game_type = state.get("game_type")
    if game_type in ("tournament", "cash-game"):
        return {"game_type": game_type}

    guessed = _heuristic(q)
    if guessed:
        return {"game_type": guessed}

    # If heuristic can't decide, DON'T guess with LLM. Ask user.
    user_choice = interrupt(
        "Is this about **tournament rules** or **cash game rules**? "
        "Reply: 'tournament' or 'cash-game'."
    )
    return {"game_type": _normalize(str(user_choice)) or "cash-game"}
