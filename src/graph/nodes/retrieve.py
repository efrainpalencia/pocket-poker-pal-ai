from typing import Any, Dict
from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    question = state['question']
    game_type = state['game_type']

    if game_type == "cash-game":
        print("---RETRIEVE CASH-GAME RULES---")
        documents = retriever.invoke(question, k=6, filter={
                                     "source": "./assets/docs/Poker Rule Book Procedures 5th Revised FINAL.pdf"})
        return {"documents": documents, "question": question}
    elif game_type == "tournament":
        print("---RETRIEVE TOURNAMENT RULES---")
        documents = retriever.invoke(question, k=6, filter={
            "source": "./assets/docs/2024_Poker_TDA_Rules_PDF_Longform_Vers_1.0_FINAL.pdf"})
        return {"documents": documents, "question": question}
    else:
        raise ValueError(f"Unknown game_type: {game_type}")
