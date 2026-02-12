from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from graph.llm.factory import get_chat_llm

load_dotenv()

# Initialize LLM
classifier_llm = get_chat_llm(temperature=0, streaming=False)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify which ruleset the user is asking about.\n"
            "Return ONLY one token: tournament, cash-game, or unknown.\n"
            "Use tournament for questions about: tournaments, blinds/antes, bagging, levels, late reg, ICM, payouts.\n"
            "Use cash-game for: rake, table stakes, buy-ins, must-move, time rake, seat changes, straddles.\n"
            "If you cannot tell, return unknown.",
        ),
        ("human", "{question}"),
    ]
)

classifier_chain = prompt | classifier_llm | StrOutputParser()
