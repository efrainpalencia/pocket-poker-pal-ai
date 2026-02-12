from typing import Any, Dict, List

from langchain_core.documents import Document

from graph.consts import SEMINOLE_NAMESPACE
from graph.state import GraphState
from graph.vectorstore import vectorstore

# ---------------------------
# Retriever
# ---------------------------


def get_retriever(
    vectorstore,
    namespace: str,
    k: int = 6,
    meta_filter: dict | None = None,
):
    """Return a retriever wrapper for the provided vectorstore.

    Args:
        vectorstore: Vector store instance exposing `as_retriever`.
        namespace: Namespace to search within.
        k: Number of results to return.
        meta_filter: Optional metadata filter mapping.

    Returns:
        A retriever object with the requested search kwargs configured.
    """

    search_kwargs = {"k": k, "namespace": namespace}
    if meta_filter:
        search_kwargs["filter"] = meta_filter
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def _retrieval_strength(docs: List[Document]) -> float:
    """Estimate retrieval strength based on total characters in docs.

    Returns a float in [0.0, 1.0] indicating how much content was
    retrieved; heuristic used for fallback selection.
    """

    if not docs:
        return 0.0
    total_chars = sum(len(d.page_content or "") for d in docs)
    return min(1.0, total_chars / 3000.0)


def stable_doc_key(d: Document) -> str:
    """Deterministic key for a Document used for de-duplication.

    Uses namespace, page, block_id and chunk_index to identify documents.
    """

    md = d.metadata or {}
    return f"{md.get('namespace')}|{md.get('page')}|{md.get('block_id')}|{md.get('chunk_index')}"


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Graph node: retrieve documents for the question in state.

    Uses configured `namespace` and optional `retrieval_strategy` to
    perform a primary retrieval and (optionally) a fallback search when
    retrieval strength is low.
    """

    question = state["question"]
    namespace = state.get("namespace")
    meta_filter = state.get("meta_filter") or {}

    # Optional: allow retry node to widen search without changing retrieve later
    strategy = state.get("retrieval_strategy") or {}
    k = int(strategy.get("k", 6))
    # default True for tournament safety
    use_fallback = bool(strategy.get("use_fallback", True))

    if not namespace:
        return {
            **state,
            "documents": [],
            "retrieval_strength": 0.0,
        }

    primary = get_retriever(
        vectorstore, namespace=namespace, k=k, meta_filter=meta_filter
    )
    docs = primary.invoke(question)

    strength = _retrieval_strength(docs)

    # Tournament fallback: if primary retrieval is weak, try Seminole Section J
    if use_fallback and state.get("game_type") == "tournament" and strength < 0.35:
        fallback = get_retriever(
            vectorstore,
            namespace=SEMINOLE_NAMESPACE,
            k=max(4, k // 2),
            meta_filter={"game_type": "tournament"},
        )
        fallback_docs = fallback.invoke(question)

        # merge + dedupe
        seen = set()
        merged = []
        for d in docs + fallback_docs:
            key = stable_doc_key(d)
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)

        docs = merged
        strength = max(strength, _retrieval_strength(fallback_docs))

    return {
        **state,
        "documents": docs,
        "retrieval_strength": strength,
    }
