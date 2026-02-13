from typing import List, Optional
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore


def run_debug_query(
    vectorstore: PineconeVectorStore,
    question: str,
    namespace: str,
    k: int = 6,
    meta_filter: Optional[dict] = None,
) -> List[Document]:

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k,
            "namespace": namespace,
            **({"filter": meta_filter} if meta_filter else {}),
        }
    )

    return retriever.invoke(question)
