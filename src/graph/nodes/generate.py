from typing import Dict, Any, List
from langchain_core.documents import Document

from graph.state import GraphState
from graph.chains.generation import generation_chain, GenerationOut


def _format_context(docs: List[Document], max_docs: int = 6) -> str:
    chunks = []
    for i, d in enumerate(docs[:max_docs], 1):
        md = d.metadata or {}
        source = md.get("source_pdf") or md.get(
            "rulebook") or md.get("source_file") or "source"
        page = md.get("page")
        section = md.get("section") or md.get("block_id") or ""
        label = " | ".join(
            [p for p in [source, section, f"p.{page}" if page is not None else ""] if p])
        text = (d.page_content or "").strip()
        if text:
            chunks.append(f"[{i}] {label}\n{text}")
    return "\n\n".join(chunks)


def _render_generation(out: GenerationOut) -> str:
    # Keep a human-readable string for UI/logging
    if out.mode == "direct":
        return f"Answer: {out.answer}\nQuote: \"{out.quote or ''}\""
    if out.mode == "inference":
        return f"Inference: {out.answer}\nQuote: \"{out.quote or ''}\"\nCaveat: {out.caveat or ''}"
    return f"{out.answer}\nClarifying Question: {out.clarifying_question or ''}"


def generate(state: GraphState) -> Dict[str, Any]:
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
