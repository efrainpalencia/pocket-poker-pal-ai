from typing import Any, Dict, List

from langchain_core.documents import Document

from graph.chains.generation import GenerationOut, generation_chain
from graph.state import GraphState


def _format_context(docs: List[Document], max_docs: int = 6) -> str:
    """Format a list of Documents into a concise context string.

    Keeps up to `max_docs` and includes a short label for each source.
    """

    chunks = []
    for i, d in enumerate(docs[:max_docs], 1):
        md = d.metadata or {}
        source = (
            md.get("source_pdf")
            or md.get("rulebook")
            or md.get("source_file")
            or "source"
        )
        page = md.get("page")
        section = md.get("section") or md.get("block_id") or ""
        label = " | ".join(
            [p for p in [source, section, f"p.{page}" if page is not None else ""] if p]
        )
        text = (d.page_content or "").strip()
        if text:
            chunks.append(f"[{i}] {label}\n{text}")
    return "\n\n".join(chunks)


def _render_generation(out: GenerationOut) -> str:
    """Render a `GenerationOut` into a human-readable string.

    This helper is used to produce the `generation` field stored in
    the graph state for logging and non-structured UIs.
    """

    # Keep a human-readable string for UI/logging
    if out.mode == "direct":
        return f"Answer: {out.answer}\nQuote: \"{out.quote or ''}\""
    if out.mode == "inference":
        return f"Inference: {out.answer}\nQuote: \"{out.quote or ''}\"\nCaveat: {out.caveat or ''}"
    return f"{out.answer}\nClarifying Question: {out.clarifying_question or ''}"


def generate(state: GraphState) -> Dict[str, Any]:
    """Graph node: render a generation for the current state.

    Invokes the `generation_chain` with the assembled context and
    returns structured and rendered outputs to be stored in state.
    """

    print("---GENERATE---")
    docs = state.get("documents") or []
    context = _format_context(docs) or "[NO_CONTEXT_RETRIEVED]"

    out: GenerationOut = generation_chain.invoke(
        {"question": state["question"], "context": context}
    )

    return {
        "generation_structured": out.model_dump(),
        # keeps your existing workflow compatible
        "generation": _render_generation(out),
        "context_used": context,
    }
