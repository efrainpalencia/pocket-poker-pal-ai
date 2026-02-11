from typing import Any, Dict
from graph.state import GraphState
from graph.chains.grader import grader_chain


def grade(state: GraphState) -> Dict[str, Any]:
    print("---GRADE---")
    result = grader_chain.invoke({
        "question": state["question"],
        "context": "\n\n".join(state.get("documents", [])),
        "answer": state.get("generation", ""),
    })

    # result is GradeOut (pydantic model)
    confidence = float(result.confidence)
    grounded = confidence >= 0.60 and not result.is_hallucination_risk

    return {
        "grade_label": result.label,
        "confidence": confidence,
        "grounded": grounded,
        "missing_info": result.missing_info,
    }
