import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# System prompt

system = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \n\n
    Question: {question} \n
    Context: {context} \n
    Answer:"""

# Chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: \\n{question}")
    ]
)

generation_chain = prompt_template | llm | StrOutputParser()
