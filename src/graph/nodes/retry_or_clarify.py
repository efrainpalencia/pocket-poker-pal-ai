from langgraph.types import interrupt
from graph.state import GraphState


def retry_or_clarify(state: GraphState) -> dict:
    attempts = int(state.get("attempts", 0)) + 1
    print(f"---RETRY_OR_CLARIFY (attempt {attempts})---")

    # 1 retry: just bump attempts and let retrieve run again (with larger k)
    if attempts <= 1:
        return {"attempts": attempts}

    # after retry, ask for more context (NOT tournament vs cash-game)
    extra = interrupt(
        "I couldn’t confirm the answer from the retrieved rule text. "
        "Can you add a bit more detail about the situation? "
        "For example: 'all-in and called at showdown', 'uncalled all-in', "
        "'showing cards', 'mucked hand', etc."
    )

    new_q = state["question"] + f"\n\nUser clarification: {extra}"
    return {"attempts": attempts, "question": new_q}
