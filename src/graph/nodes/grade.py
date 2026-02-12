from graph.state import GraphState
from graph.chains.grader import grade_answer


def grade(state: GraphState) -> dict:
    context = state.get("context_used") or ""
    question = state.get("question", "") or ""
    generation = state.get("generation", "") or ""

    gs = state.get("generation_structured")

    # Pull structured fields if present
    mode = None
    quote = None
    clarifying_q = None

    if isinstance(gs, dict):
        mode = (gs.get("mode") or "").strip() or None
        quote = (gs.get("quote") or "").strip() or None
        clarifying_q = (gs.get("clarifying_question") or "").strip() or None

    # ✅ One grading path: deterministic + LLM inside grade_answer
    result = grade_answer(
        question=question,
        context=context,
        answer=generation,
        quote=quote,
        mode=mode,
        clarifying_question=clarifying_q,
    )

    confidence = float(result.confidence)
    hallucination = bool(getattr(result, "is_hallucination_risk", False))
    grounded = confidence >= 0.60 and not hallucination

    return {
        "grade_label": result.label,
        "confidence": confidence,
        "grounded": grounded,
        "missing_info": result.missing_info,
        "is_hallucination_risk": hallucination,
    }
