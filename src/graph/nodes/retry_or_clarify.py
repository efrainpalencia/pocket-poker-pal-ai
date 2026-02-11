from langgraph.types import interrupt
from graph.state import GraphState


def retry_or_clarify(state: GraphState) -> dict:
    missing = state.get("missing_info") or []
    if missing:
        prompt = (
            "I couldn’t answer confidently from the rule text. "
            "Can you clarify: " + "; ".join(missing) + "?"
        )
    else:
        prompt = (
            "I couldn’t confirm the answer from the retrieved rule text. "
            "Can you add a bit more detail about the situation?"
        )
    user_more = interrupt(prompt)
    # You can store this as an additional context / refinement field if you want
    return {"question": f"{state['question']}\n\nExtra detail: {user_more}"}
