from fastapi import APIRouter, HTTPException

from schemas.chat_schema import AskIn, ResumeIn, QAOut

from services.chat_service import get_answer, get_clarification


router = APIRouter()


@router.post("/qa", response_model=QAOut, status_code=200)
def qa(req: AskIn):
    try:
        query = get_clarification(req)
        return query
    except Exception:
        raise HTTPException(
            status_code=500, detail="Error occured while making request.")


@router.post("/qa/resume", response_model=ResumeIn, status_code=200)
def qa_resume(req: ResumeIn):
    try:
        query = get_answer(req)
        return query
    except Exception:
        raise HTTPException(
            status_code=500, detail="Error occured while making request.")
