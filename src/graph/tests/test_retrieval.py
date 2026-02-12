import os
import pytest
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from graph.retrieval_debug import run_debug_query

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME")


@pytest.fixture
def vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)


def test_tournament_dead_small_blind_retrieval(vectorstore):
    """
    Ensure tournament namespace retrieves relevant chunk
    for 'dead small blind' question.
    """

    question = "dead small blind new player assume small blind"

    docs = run_debug_query(
        vectorstore=vectorstore,
        question=question,
        namespace="tournament_tda_2024",
    )

    # 1️⃣ We should get at least one document
    assert len(docs) > 0, "No documents retrieved from tournament namespace"

    # 2️⃣ At least one doc should mention small blind or blind posting
    combined_text = " ".join(d.page_content.lower() for d in docs)

    assert (
        "small blind" in combined_text
        or "blind" in combined_text
    ), "Retrieved docs do not appear relevant to blinds"


def test_seminole_section_j_tournament_filter(vectorstore):
    """
    Ensure Seminole namespace retrieves tournament section content
    when filtered.
    """

    question = "dead small blind new player assume small blind"

    docs = run_debug_query(
        vectorstore=vectorstore,
        question=question,
        namespace="house_seminole_2025",
        meta_filter={"game_type": "tournament"},
    )

    # 1️⃣ Ensure docs were returned
    assert len(docs) > 0, "No documents retrieved from Seminole tournament section"

    # 2️⃣ Ensure at least one chunk comes from Section_J
    sections = [d.metadata.get("section") for d in docs]

    assert (
        "Section_J" in sections
    ), f"Section_J not found in retrieved metadata. Sections returned: {sections}"
