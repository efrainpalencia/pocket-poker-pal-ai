from langgraph.types import interrupt

from graph.consts import SEMINOLE_NAMESPACE, TDA_NAMESPACE
from graph.state import GraphState

MAX_RETRIES = 2


def _is_ruleset_missing(state: GraphState) -> bool:
    """Return True when a ruleset/game_type or namespace is missing.

    Used to decide whether to ask the user to choose tournament vs cash-game.
    """

    return (
        bool(state.get("needs_clarification"))
        or not state.get("game_type")
        or not state.get("namespace")
    )


def retry_or_clarify(state: GraphState) -> dict:
    """Graph node: either request clarifying information or prepare a retry.

    Handles two main cases:
      - Missing ruleset (tournament vs cash-game) — ask user to choose.
      - Low-confidence answer — prompt user to provide more detail or
        append user detail and retry the retrieval/generation loop.
    """

    retry_count = int(state.get("retry_count", 0))
    missing = state.get("missing_info") or []

    # ---- Case A: UI chooses tournament vs cash-game ----
    if _is_ruleset_missing(state):
        choice = interrupt(
            {
                "type": "choose_ruleset",
                "message": "I need to know which ruleset to use for your question.",
                "options": ["tournament", "cash-game"],
            }
        )

        selected = (choice or "").strip().lower()

        if selected in {"tournament", "tourney"}:
            return {
                # ✅ reset is OK here: this is routing setup, not a “retry loop”
                "retry_count": 0,
                "needs_clarification": False,
                "missing_info": [],
                "game_type": "tournament",
                "namespace": TDA_NAMESPACE,
                "meta_filter": {},
            }

        if selected in {"cash-game", "cash game", "cash"}:
            return {
                "retry_count": 0,
                "needs_clarification": False,
                "missing_info": [],
                "game_type": "cash-game",
                "namespace": SEMINOLE_NAMESPACE,
                "meta_filter": {"game_type": "cash-game"},
            }

        return {
            "retry_count": retry_count + 1,
            "needs_clarification": True,
            "missing_info": ["Please choose: tournament or cash-game."],
        }

    # ---- Case B: low-confidence loop ----
    # ✅ If we've already asked enough times, stop looping.
    if retry_count >= MAX_RETRIES:
        return {"force_end": True}

    if missing:
        prompt = (
            " ".join(missing)
            if any("?" in m for m in missing)
            else (
                "I couldn’t answer confidently from the rule text. Can you clarify: "
                + "; ".join(missing)
                + "?"
            )
        )
    else:
        prompt = (
            "I couldn’t confirm the answer from the retrieved rule text. "
            "Can you add a bit more detail about the situation?"
        )

    user_more = interrupt({"type": "free_text", "message": prompt})

    return {
        "retry_count": retry_count + 1,
        "needs_clarification": False,
        "missing_info": [],
        "question": f"{state['question']}\n\nExtra detail: {user_more}",
        "user_extra_detail": user_more,
    }
