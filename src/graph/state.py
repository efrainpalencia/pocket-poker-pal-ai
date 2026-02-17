from typing import Any, Dict, List, Literal, NotRequired, TypedDict


class GraphState(TypedDict):
    question: str

    # ----------------
    # routing
    # ----------------
    needs_clarification: NotRequired[bool]
    missing_info: NotRequired[List[str]]

    # current routing used for retrieval
    game_type: NotRequired[Literal["tournament", "cash-game"] | str]
    namespace: NotRequired[str]
    meta_filter: NotRequired[Dict[str, Any]]

    # routing intelligence (new)
    inferred_game_type: NotRequired[Literal["tournament", "cash-game", "unknown"]]
    last_game_type: NotRequired[Literal["tournament", "cash-game"] | str]
    # True when user explicitly chooses ruleset
    routing_locked: NotRequired[bool]
    # prevents ping-pong (try other namespace once)
    fallback_attempted: NotRequired[bool]

    # optional: keep the last prompt the UI can render
    # e.g. {"type":"choose_ruleset",...}
    prompt: NotRequired[Dict[str, Any]]

    # ----------------
    # retrieval
    # ----------------
    documents: NotRequired[Any]  # list[Document] or list[str]
    retrieval_strength: NotRequired[float]
    retrieval_strategy: NotRequired[Dict[str, Any]]

    # ----------------
    # generation
    # ----------------
    generation: NotRequired[str]
    context_used: NotRequired[str]
    generation_structured: NotRequired[Dict[str, Any]]

    # ----------------
    # grading
    # ----------------
    confidence: NotRequired[float]
    grounded: NotRequired[bool]

    # ----------------
    # retry control
    # ----------------
    retry_count: NotRequired[int]
    force_end: NotRequired[bool]
