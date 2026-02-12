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


system = """You are a poker rules assistant.

Use ONLY the provided context. Do not use outside knowledge.

Choose exactly one mode:
- direct: the context explicitly states the rule/answer
- inference: the context strongly implies the answer, consistent with common poker procedure
- not_found: the context is insufficient to answer

STRICT OUTPUT REQUIREMENTS:
- mode=direct:
  - answer: max 4 sentences
  - quote: REQUIRED. Must be an EXACT contiguous excerpt copied from the context.
  - quote must be <= 25 words (count words by spaces).
  - caveat MUST be null/omitted
  - clarifying_question MUST be null/omitted
- mode=inference:
  - answer: max 4 sentences, start with "Inference: "
  - quote: REQUIRED. Must be an EXACT contiguous excerpt copied from the context (<= 25 words)
  - caveat: REQUIRED (1 sentence)
  - clarifying_question MUST be null/omitted
- mode=not_found:
  - answer MUST be exactly: "Not found in the provided text."
  - clarifying_question: REQUIRED (exactly one question)
  - quote MUST be null/omitted
  - caveat MUST be null/omitted

Never invent rule numbers, penalties, amounts, or citations.
If unsure, prefer mode=not_found.
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
