from typing import Literal
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    question: str
    game_type: Literal["tournament", "cash-game"] | None
    documents: list[str]
    generation: str
    grounded: bool
