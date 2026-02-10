from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]

    # documents should be list[str] in state
    docs = state.get("documents", [])
    if not isinstance(docs, list):
        docs = [str(docs)]

    # Join into a single context string for the LLM
    context = "\n\n".join(str(d) for d in docs if d)

    # Optional: helpful debug during dev
    # print("CONTEXT_CHARS:", len(context))

    result = generation_chain.invoke(
        {"context": context, "question": question})

    # Normalize to plain string
    generation_text = getattr(result, "content", result)
    if not isinstance(generation_text, str):
        generation_text = str(generation_text)

    return {"generation": generation_text}
