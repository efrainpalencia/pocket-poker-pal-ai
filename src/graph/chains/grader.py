import re
from typing import List, Literal, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.llm.factory import get_chat_llm

load_dotenv()


class GradeOut(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    label: Literal["YES", "PARTIAL", "NO"]
    reasons: List[str] = []
    missing_info: List[str] = []
    is_hallucination_risk: bool = False


# -------------------------
# Deterministic helpers
# -------------------------

_NOT_FOUND_PREFIX = "Not found in the provided text."
QUOTE_RE = re.compile(r'(?im)^\s*Quote:\s*"?(.*?)"?\s*$', re.MULTILINE)
CLARIFY_RE = re.compile(r"(?im)^\s*Clarifying Question:\s*(.+)\s*$", re.MULTILINE)


def extract_quote(answer: str) -> Optional[str]:
    """Extract the 'Quote:' line from a generated answer, if present.

    Returns the quote text or `None` when not found.
    """

    if not answer:
        return None
    m = QUOTE_RE.search(answer)
    if not m:
        return None
    q = (m.group(1) or "").strip()
    return q or None


def extract_clarifying_question(answer: str) -> Optional[str]:
    """Extract an inline 'Clarifying Question:' from an answer string.

    Returns the clarifying question text or `None` when not present.
    """

    if not answer:
        return None
    m = CLARIFY_RE.search(answer)
    if not m:
        return None
    s = (m.group(1) or "").strip()
    return s or None


def quote_in_context(quote: str, context: str) -> bool:
    """Check whether a quoted excerpt appears in the provided context.

    Normalizes whitespace for robust matching.
    """

    if not quote or not context:
        return False

    def norm(s: str) -> str:
        return " ".join(s.split())

    return norm(quote) in norm(context)


def robust_quote_in_context(quote: str, context: str) -> bool:
    """More permissive quote matching.

    This function normalizes whitespace, punctuation, and common
    unicode quote characters before checking containment. It helps
    avoid false negatives when PDF-extracted text contains odd
    punctuation or non-breaking spaces.
    """

    if not quote or not context:
        return False

    import re

    def normalize(s: str) -> str:
        # replace various unicode quote characters with plain quote
        s = s.replace("\u201c", '"').replace("\u201d", '"')
        s = s.replace("\u2018", "'").replace("\u2019", "'")
        # collapse whitespace
        s = " ".join(s.split())
        # remove punctuation except alphanumerics and spaces
        s = re.sub(r"[^0-9A-Za-z\s]", "", s)
        return s.lower()

    nq = normalize(quote)
    nc = normalize(context)
    return nq in nc


# -------------------------
# LLM grader chain
# -------------------------

parser = PydanticOutputParser(pydantic_object=GradeOut)
grader_llm = get_chat_llm(temperature=0, streaming=False)

grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a poker rules answer evaluator.\n\n"
            "Evaluate whether the answer is supported by the provided context.\n"
            "The assistant MAY make reasonable procedural inferences when the context does not state "
            "the answer verbatim, as long as the inference is consistent with the context and common poker procedure.\n\n"
            "Scoring guidance (0.0 to 1.0):\n"
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
            "missing_info (array of strings), is_hallucination_risk (boolean).",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:\n{answer}\n\n"
            "JSON:",
        ),
    ]
)

grader_chain = grader_prompt | grader_llm | parser


# -------------------------
# Public grading function
# -------------------------


def grade_answer(
    question: str,
    context: str,
    answer: str,
    *,
    quote: Optional[str] = None,
    mode: Optional[str] = None,
    clarifying_question: Optional[str] = None,
) -> GradeOut:
    """
    Combines deterministic checks with LLM grading.

    If you have structured generation, pass quote/mode/clarifying_question
    so deterministic grounding checks still work.
    """
    answer = answer or ""
    context = context or ""

    # Prefer structured quote if provided
    if quote:
        quote = quote.strip()
    else:
        quote = extract_quote(answer)

    # Prefer structured clarifying question if provided
    if not clarifying_question:
        clarifying_question = extract_clarifying_question(answer)

    # 1) Hard fail if not_found (structured or string prefix)
    if (mode == "not_found") or answer.strip().startswith(_NOT_FOUND_PREFIX):
        missing = [clarifying_question] if clarifying_question else []
        return GradeOut(
            confidence=0.0,
            label="NO",
            reasons=["Model indicated the answer was not found in the provided text."],
            missing_info=missing,
            is_hallucination_risk=False,
        )

    # 2) If quote exists try to verify it appears in context. Prefer a
    # strict match but fall back to a more permissive check so small
    # formatting/encoding differences don't falsely mark answers as
    # hallucinated. If both checks fail, mark `quote_mismatch` and
    # continue to the LLM grader rather than failing immediately.
    quote_mismatch = False
    if quote and not quote_in_context(quote, context):
        if not robust_quote_in_context(quote, context):
            quote_mismatch = True

    # 2b) If mode says direct/inference but quote is missing => fail fast
    if mode in {"direct", "inference"} and not quote:
        return GradeOut(
            confidence=0.0,
            label="NO",
            reasons=[
                "Structured mode requires a supporting quote, but none was provided."
            ],
            missing_info=[],
            is_hallucination_risk=True,
        )

    # 3) Run LLM grader
    llm_result: GradeOut = grader_chain.invoke(
        {"question": question, "context": context, "answer": answer}
    )

    # 4) Cap inference confidence (structured mode OR string prefix)
    is_inference = (mode == "inference") or answer.lstrip().startswith("Inference:")
    if is_inference:
        llm_result.confidence = float(min(llm_result.confidence, 0.84))
        if llm_result.confidence >= 0.60:
            llm_result.label = "PARTIAL"
        else:
            llm_result.label = "NO"

    # If we detected a quote mismatch, reduce trust in the grader output
    # and force hallucination risk to True. This avoids giving a high
    # confidence score when the quoted evidence cannot be robustly
    # located in the retrieved context.
    if quote_mismatch:
        llm_result.confidence = float(min(llm_result.confidence, 0.84))
        llm_result.is_hallucination_risk = True

    # 5) Ensure label matches thresholds
    c = float(llm_result.confidence)
    if c >= 0.85:
        llm_result.label = "YES"
    elif c >= 0.60:
        llm_result.label = "PARTIAL"
    else:
        llm_result.label = "NO"

    return llm_result
