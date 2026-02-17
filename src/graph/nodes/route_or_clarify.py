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
    q = state.get("question", "") or ""

    existing_game_type = state.get("game_type")
    existing_namespace = state.get("namespace")
    routing_locked = bool(state.get("routing_locked", False))

    # 1) Infer for THIS question
    inferred = heuristic_classify(q)
    inferred_is_strong = inferred in {"tournament", "cash-game"}

    if inferred == "unknown":
        raw = classifier_chain.invoke({"question": q})
        inferred = normalize_game_type(raw)
        # treat classifier result as weaker than heuristic
        inferred_is_strong = False if inferred == "unknown" else True

    # 2) If user explicitly chose ruleset earlier, keep it unless question is strongly opposite
    if (
        routing_locked
        and existing_game_type in {"tournament", "cash-game"}
        and existing_namespace
    ):
        if (
            inferred in {"tournament", "cash-game"}
            and inferred != existing_game_type
            and inferred_is_strong
        ):
            routed = _routing_for(inferred)
            routed["inferred_game_type"] = inferred
            return routed

        routed = _routing_for(existing_game_type)
        routed["inferred_game_type"] = (
            inferred if inferred != "unknown" else existing_game_type
        )
        return routed

    # 3) If we have existing routing but it's not locked, allow auto-switch when inferred differs
    if existing_game_type in {"tournament", "cash-game"} and existing_namespace:
        if inferred in {"tournament", "cash-game"} and inferred != existing_game_type:
            # auto-switch (no prompt) because user didn't explicitly lock routing
            routed = _routing_for(inferred)
            routed["inferred_game_type"] = inferred
            return routed

        routed = _routing_for(existing_game_type)
        routed["inferred_game_type"] = existing_game_type
        return routed

    # 4) No existing routing -> use inferred or ask
    if inferred in {"tournament", "cash-game"}:
        routed = _routing_for(inferred)
        routed["inferred_game_type"] = inferred
        return routed

    return {
        "needs_clarification": True,
        "missing_info": ["Is this about tournament rules or cash-game rules?"],
        "prompt": {
            "type": "choose_ruleset",
            "message": "I need to know which ruleset to use for your question.",
            "options": ["tournament", "cash-game"],
        },
    }
