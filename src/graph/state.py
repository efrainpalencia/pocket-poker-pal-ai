from typing import TypedDict, NotRequired, List, Any, Dict


class GraphState(TypedDict):
    question: str

    # routing
    needs_clarification: NotRequired[bool]
    missing_info: NotRequired[List[str]]
    game_type: NotRequired[str]
    namespace: NotRequired[str]
    meta_filter: NotRequired[Dict[str, Any]]

    # retrieval
    documents: NotRequired[Any]  # list[Document] or list[str]
    retrieval_strength: NotRequired[float]
    retrieval_strategy: NotRequired[Dict[str, Any]]

    # generation
    generation: NotRequired[str]
    context_used: NotRequired[str]
    generation_structured: NotRequired[Dict[str, Any]]

    # grading
    confidence: NotRequired[float]
    grounded: NotRequired[bool]

    # retry control
    retry_count: NotRequired[int]
    force_end: NotRequired[bool]
