from typing import Any, Dict, List
from langchain_core.documents import Document

from graph.state import GraphState
from graph.consts import SEMINOLE_NAMESPACE
from graph.vectorstore import vectorstore  # ✅ singleton


# ---------------------------
# Retriever
# ---------------------------

def get_retriever(
    vectorstore,
    namespace: str,
    k: int = 6,
    meta_filter: dict | None = None,
):
    search_kwargs = {"k": k, "namespace": namespace}
    if meta_filter:
        search_kwargs["filter"] = meta_filter
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def _retrieval_strength(docs: List[Document]) -> float:
    if not docs:
        return 0.0
    total_chars = sum(len(d.page_content or "") for d in docs)
    return min(1.0, total_chars / 3000.0)


def stable_doc_key(d: Document) -> str:
    md = d.metadata or {}
    return f"{md.get('namespace')}|{md.get('page')}|{md.get('block_id')}|{md.get('chunk_index')}"


def retrieve(state: GraphState) -> Dict[str, Any]:
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
        vectorstore, namespace=namespace, k=k, meta_filter=meta_filter)
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
        for d in (docs + fallback_docs):
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
