from pydantic import BaseModel


class AskIn(BaseModel):
    question: str
    thread_id: str | None = None


class ResumeIn(BaseModel):
    thread_id: str
    reply: str


class QAOut(BaseModel):
    status: str
    thread_id: str
    prompt: str | None = None
    generation: str | None = None
