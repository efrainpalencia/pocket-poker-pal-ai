from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    result = generation_chain.invoke(
        {"context": documents, "question": question})

    # If it's an AIMessage, it will have .content
    generation_text = getattr(result, "content", result)
    if not isinstance(generation_text, str):
        generation_text = str(generation_text)

    # Return only updates (deltas)
    return {"generation": generation_text}
