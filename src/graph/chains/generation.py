import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from graph.llm.factory import get_chat_llm

load_dotenv()

# Initialize LLM
llm = get_chat_llm(temperature=0, streaming=True)

# System prompt

system = """You are a poker rules assistant.
Answer using ONLY the retrieved context below.

Rules:
- If the context contains the rule: answer and include a short supporting quote (max 25 words).
- If the rule is not explicitly stated but can be reasonably inferred: say "Inference:" and include the closest supporting quote + a short caveat.
- If the context is insufficient: say "Not found in the provided text" and ask ONE clarifying question.

Keep it concise (max 4 sentences).

Question: {question}
Context: {context}
Answer:"""


# Chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: \\n{question}")
    ]
)

generation_chain = prompt_template | llm | StrOutputParser()
