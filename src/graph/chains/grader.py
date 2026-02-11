import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Literal, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from graph.llm.factory import get_chat_llm

load_dotenv()


class GradeOut(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    label: Literal["YES", "PARTIAL", "NO"]
    reasons: List[str] = []
    missing_info: List[str] = []
    is_hallucination_risk: bool = False


# Initialize LLM
parser = PydanticOutputParser(pydantic_object=GradeOut)
llm = get_chat_llm(temperature=0, streaming=False)

grader_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a poker rules answer evaluator.\n\n"
     "Evaluate whether the answer is supported by the provided context.\n"
     "The assistant MAY make reasonable procedural inferences when the context does not state "
     "the answer verbatim, as long as the inference is consistent with the context and common poker procedure.\n\n"
     "Score confidence from 0.0 to 1.0:\n"
     "- 0.85–1.00: Strongly supported or clearly implied (safe)\n"
     "- 0.60–0.84: Reasonable inference, some uncertainty (still usable)\n"
     "- 0.00–0.59: Weak/unsupported, likely wrong or missing key facts\n\n"
     "Label rules:\n"
     "- YES if confidence >= 0.85\n"
     "- PARTIAL if 0.60 <= confidence < 0.85\n"
     "- NO if confidence < 0.60\n\n"
     "Hallucination risk is TRUE if the answer invents specific rules, penalties, amounts, or citations "
     "not implied by context.\n\n"
     "Return ONLY valid JSON with keys:\n"
     "confidence (number), label (YES|PARTIAL|NO), reasons (array of strings), "
     "missing_info (array of strings), is_hallucination_risk (boolean)."
     ),
    ("human",
     "Question:\n{question}\n\n"
     "Context:\n{context}\n\n"
     "Answer:\n{answer}\n\n"
     "JSON:"
     )
])
grader_llm = ChatOpenAI(temperature=0)
grader_chain = grader_prompt | grader_llm | parser
