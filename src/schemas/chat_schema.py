from typing import Literal, Optional, List, Union
from pydantic import BaseModel, Field


# ---------- interrupt payloads ----------

class ChooseRulesetPrompt(BaseModel):
    type: Literal["choose_ruleset"]
    message: str
    options: List[Literal["tournament", "cash-game"]]


class FreeTextPrompt(BaseModel):
    type: Literal["free_text"]
    message: str


InterruptPrompt = Union[ChooseRulesetPrompt, FreeTextPrompt]


# ---------- SSE event models (optional but nice) ----------

class SSEStart(BaseModel):
    type: Literal["start"]
    thread_id: str


class SSEToken(BaseModel):
    type: Literal["token"]
    thread_id: str
    text: str


class SSENeedsClarification(BaseModel):
    type: Literal["needs_clarification"]
    thread_id: str
    prompt: InterruptPrompt


class SSEComplete(BaseModel):
    type: Literal["complete"]
    thread_id: str
    generation: str


class SSEError(BaseModel):
    type: Literal["error"]
    thread_id: str
    message: str


# ---------- request bodies (if you later add POST endpoints) ----------

class AskIn(BaseModel):
    question: str = Field(min_length=1)
    thread_id: Optional[str] = None


class ResumeIn(BaseModel):
    thread_id: str = Field(min_length=1)
    # UI sends either "tournament" or free text
    reply: str = Field(min_length=1)


# ---------- non-streaming output (optional) ----------

class QAOut(BaseModel):
    status: Literal["needs_clarification", "complete"]
    thread_id: str
    prompt: Optional[InterruptPrompt] = None
    generation: Optional[str] = None
