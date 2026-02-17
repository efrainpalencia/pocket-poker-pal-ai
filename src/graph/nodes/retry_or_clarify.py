from __future__ import annotations

from langgraph.types import interrupt

from graph.consts import SEMINOLE_NAMESPACE, TDA_NAMESPACE
from graph.state import GraphState

MAX_RETRIES = 2


def _normalize_ruleset(raw: str | None) -> str:
    s = (raw or "").strip().lower()
    if s in {"tournament", "tourney"}:
        return "tournament"
    if s in {"cash-game", "cash game", "cash", "cashgame"}:
        return "cash-game"
    return "unknown"


def _routing_for(game_type: str) -> dict:
    if game_type == "tournament":
        return {"game_type": "tournament", "namespace": TDA_NAMESPACE, "meta_filter": {}}
    if game_type == "cash-game":
        return {
            "game_type": "cash-game",
            "namespace": SEMINOLE_NAMESPACE,
            "meta_filter": {"game_type": "cash-game"},
        }
    return {}


def _opposite(game_type: str) -> str:
    return "cash-game" if game_type == "tournament" else "tournament"


def _needs_ruleset_prompt(state: GraphState) -> bool:
    """
    We should ask for ruleset when:
    - route_or_clarify set needs_clarification True, OR
    - game_type/namespace missing
    """
    return bool(state.get("needs_clarification")) or not state.get("game_type") or not state.get("namespace")


def _build_ruleset_prompt(state: GraphState) -> dict:
    """
    If we have an existing game_type, this is a reconfirm prompt.
    Otherwise it’s the initial ruleset selection prompt.
    """
    existing = state.get("game_type")
    if existing in {"tournament", "cash-game"}:
        other = _opposite(existing)
        return {
            "type": "choose_ruleset",
            "message": f"You previously selected {existing}. Is this question about {existing} or {other}?",
            "options": [existing, other],
        }

    return {
        "type": "choose_ruleset",
        "message": "I need to know which ruleset to use for your question.",
        "options": ["tournament", "cash-game"],
    }


def _append_extra_detail(original: str, extra: str) -> str:
    extra = (extra or "").strip()
    if not extra:
        return original
    return f"{original}\n\nExtra detail: {extra}"


def retry_or_clarify(state: GraphState) -> dict:
    """
    Graph node: either request clarifying information or prepare a retry.

    Responsibilities:
      A) ruleset selection / reconfirmation (choose_ruleset)
      B) low-confidence clarification loop (free_text)
    """
    # Respect force_end if already set upstream
    if state.get("force_end"):
        return {"force_end": True}

    retry_count = int(state.get("retry_count", 0))
    missing = state.get("missing_info") or []

    # ---- Case A: Need ruleset (initial or reconfirm) ----
    if _needs_ruleset_prompt(state):
        prompt = state.get("prompt")
        if not isinstance(prompt, dict) or prompt.get("type") != "choose_ruleset":
            prompt = _build_ruleset_prompt(state)

        choice = interrupt(prompt)
        selected = _normalize_ruleset(choice)

        if selected in {"tournament", "cash-game"}:
            return {
                # ruleset selection is routing, not a "retry" — reset retry loop
                "force_end": False,
                "retry_count": 0,
                "needs_clarification": False,
                "missing_info": [],
                "prompt": None,
                "last_game_type": selected,
                **_routing_for(selected),
            }

        # Invalid selection: ask again (DON’T burn retries)
        return {
            "force_end": False,
            "needs_clarification": True,
            "missing_info": ["Please choose: tournament or cash-game."],
            "prompt": {
                "type": "choose_ruleset",
                "message": "Please choose one option to continue.",
                "options": ["tournament", "cash-game"],
            },
        }

    # ---- Case B: Low-confidence / insufficient context loop ----
    if retry_count >= MAX_RETRIES:
        return {"force_end": True}

    # Prefer explicit missing_info messaging if provided
    if missing:
        # If any item already looks like a question, use it directly
        if any("?" in str(m) for m in missing):
            prompt_text = " ".join(str(m) for m in missing)
        else:
            prompt_text = (
                "I couldn’t answer confidently from the rule text. Can you clarify: "
                + "; ".join(str(m) for m in missing)
                + "?"
            )
    else:
        prompt_text = (
            "I couldn’t confirm the answer from the retrieved rule text. "
            "Can you add a bit more detail about the situation?"
        )

    user_more = interrupt({"type": "free_text", "message": prompt_text})
    user_more = (user_more or "").strip()

    return {
        "retry_count": retry_count + 1,
        "needs_clarification": False,
        "missing_info": [],
        "prompt": None,
        "question": _append_extra_detail(state.get("question", "") or "", user_more),
        "user_extra_detail": user_more,
    }
