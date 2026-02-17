from graph.chains.classifier import classifier_chain
from graph.consts import SEMINOLE_NAMESPACE, TDA_NAMESPACE
from graph.state import GraphState

TOURNAMENT_HINTS = [
    "tournament", "level", "levels", "late reg", "late registration",
    "registration", "bag", "bagging", "break", "icm", "payout", "ante",
    "button ante", "bb ante", "re-entry", "rebuy", "add-on", "chip race",
]

CASH_HINTS = [
    "cash game", "rake", "time rake", "must-move", "table stakes",
    "buy-in", "straddle", "missed blind", "seat change", "runner",
    "comp", "time charge",
]


def heuristic_classify(question: str) -> str:
    q = (question or "").lower()
    if any(h in q for h in TOURNAMENT_HINTS):
        return "tournament"
    if any(h in q for h in CASH_HINTS):
        return "cash-game"
    return "unknown"


def normalize_game_type(raw: str) -> str:
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


def _routing_for(game_type: str) -> dict:
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
    return {"needs_clarification": True}


def _opposite(game_type: str) -> str:
    return "cash-game" if game_type == "tournament" else "tournament"


def _ask_to_reconfirm(previous: str) -> dict:
    other = _opposite(previous)
    return {
        "needs_clarification": True,
        # ✅ clear stale routing so no downstream node accidentally uses it
        "namespace": None,
        "meta_filter": {},
        "missing_info": [f"Are you asking about {previous} rules or {other} rules?"],
        "prompt": {
            "type": "choose_ruleset",
            "message": f"You previously selected {previous}. Is this question about {previous} or {other}?",
            "options": [previous, other],
        },
        "last_game_type": previous,
    }


def _has_hint_for(game_type: str, question: str) -> bool:
    q = (question or "").lower()
    if game_type == "tournament":
        return any(h in q for h in TOURNAMENT_HINTS)
    if game_type == "cash-game":
        return any(h in q for h in CASH_HINTS)
    return False


def route_or_clarify(state: GraphState) -> dict:
    """
    - Sticky ruleset within a thread (Option 1 session behavior)
    - Reconfirm ruleset on strong evidence of a switch (Option 3)
    """
    q = state.get("question", "") or ""

    existing_game_type = state.get("game_type")
    existing_namespace = state.get("namespace")

    # 1) Heuristic classification (strong signal)
    inferred_heur = heuristic_classify(q)

    # 2) LLM classifier only if heuristic doesn't know
    inferred = inferred_heur
    inferred_source = "heuristic"
    if inferred == "unknown":
        raw = classifier_chain.invoke({"question": q})
        inferred = normalize_game_type(raw)
        inferred_source = "llm"

    # 3) If we already have routing, decide whether to keep it
    if existing_game_type in {"tournament", "cash-game"} and existing_namespace:
        # ✅ Reconfirm only on strong signal:
        # - heuristic says opposite (strong), OR
        # - llm says opposite AND question contains explicit opposite hints
        if inferred in {"tournament", "cash-game"} and inferred != existing_game_type:
            if inferred_source == "heuristic" or _has_hint_for(inferred, q):
                return _ask_to_reconfirm(existing_game_type)

        routed = _routing_for(existing_game_type)
        routed["last_game_type"] = existing_game_type
        return routed

    # 4) No existing routing -> use inferred if confident, else ask
    if inferred in {"tournament", "cash-game"}:
        routed = _routing_for(inferred)
        routed["last_game_type"] = inferred
        return routed

    # 5) Unknown -> ask
    return {
        "needs_clarification": True,
        "namespace": None,
        "meta_filter": {},
        "missing_info": ["Is this about tournament rules or cash-game rules?"],
        "prompt": {
            "type": "choose_ruleset",
            "message": "I need to know which ruleset to use for your question.",
            "options": ["tournament", "cash-game"],
        },
    }
