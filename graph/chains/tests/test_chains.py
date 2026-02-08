from ingestion import retriever
from graph.chains.generation import generation_chain
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()


def test_generation() -> None:
    question = "Does a player have to turn their cards face-up if they are all-in, but the river has not been dealt yet?"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {"context": docs, "question": question}
    )
    pprint(generation)
