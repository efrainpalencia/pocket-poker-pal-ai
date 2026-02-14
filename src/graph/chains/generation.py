from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.llm.factory import get_chat_llm

load_dotenv()


class GenerationOut(BaseModel):
    mode: Literal["direct", "inference", "not_found"] = Field(
        description="direct if explicitly supported, inference if reasonably implied, not_found if insufficient context"
    )
    answer: str = Field(
        description="Concise answer. Max 4 sentences. Do NOT include the quote here."
    )
    quote: Optional[str] = Field(
        default=None,
        description="EXACT contiguous quote copied from context (<= 25 words). Required for direct/inference.",
    )
    caveat: Optional[str] = Field(
        default=None,
        description="Short caveat about uncertainty (1 sentence). Required for inference.",
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="Ask exactly one clarifying question. Required for not_found.",
    )


system = """You are a professional poker rules assistant.

Use the provided context as your primary source.
You may rely on standard poker procedure ONLY when it aligns with the context.

Choose one mode:
- direct: the context clearly states the answer
- inference: the context implies the answer based on standard poker procedure
- partial: the context addresses part of the situation but not all variables
- not_found: the context does not sufficiently address the question

GENERAL RULES:
- Never invent rule numbers, penalties, amounts, or citations.
- If the ruling depends on circumstances, state that clearly.
- When relevant, mention that final decisions may be at floor discretion.
- If unsure, prefer "partial" over "not_found" when the context is relevant but incomplete.

OUTPUT REQUIREMENTS:

mode=direct:
- answer: max 5 sentences
- quote: REQUIRED. Exact contiguous excerpt from context (<= 35 words)
- caveat: optional (1 short sentence if discretion applies)
- clarifying_question: omitted

mode=inference:
- answer: max 5 sentences, begin with "Inference: "
- quote: REQUIRED (<= 35 words)
- caveat: REQUIRED (1 sentence explaining procedural assumption)
- clarifying_question: omitted

mode=partial:
- answer: max 5 sentences
- quote: REQUIRED (<= 35 words)
- caveat: REQUIRED (1 sentence explaining what is missing)
- clarifying_question: REQUIRED (exactly one question)

mode=not_found:
- answer: exactly "Not found in the provided text."
- clarifying_question: REQUIRED (exactly one question)
- quote: omitted
- caveat: omitted
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\nReturn structured output only.",
        ),
    ]
)

generation_llm = get_chat_llm(temperature=0, streaming=False).with_structured_output(
    GenerationOut
)

generation_chain = prompt | generation_llm
