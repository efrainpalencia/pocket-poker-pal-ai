
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Classify which ruleset the user is asking about.\n"
     "Return ONLY one token: tournament, cash-game, or unknown.\n"
     "Use tournament for questions about: tournaments, blinds/antes, bagging, levels, late reg, ICM, payouts.\n"
     "Use cash-game for: rake, table stakes, buy-ins, must-move, time rake, seat changes, straddles.\n"
     "If you cannot tell, return unknown."),
    ("human", "{question}")
])

classifier_chain = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
