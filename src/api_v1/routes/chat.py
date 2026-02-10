from fastapi import APIRouter, HTTPException
from schemas.chat_schema import AskIn, ResumeIn, QAOut
from services.chat_service import get_answer, get_clarification

router = APIRouter()


@router.post("/qa", response_model=QAOut, status_code=200)
def qa(req: AskIn):
    try:
        return get_clarification(req)
    except Exception:
        raise HTTPException(
            status_code=500, detail="Error occurred while making request.")


@router.post("/qa/resume", response_model=QAOut, status_code=200)
def qa_resume(req: ResumeIn):
    try:
        return get_answer(req)
    except Exception:
        raise HTTPException(
            status_code=500, detail="Error occurred while making request.")
