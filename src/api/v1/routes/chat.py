from fastapi import APIRouter, HTTPException
from schemas.chat_schema import AskIn, ResumeIn
from services.chat_service import ask_question, resume_question

router = APIRouter()


@router.post("/qa")
async def qa_post(payload: AskIn):
    try:
        return ask_question(
            question=payload.question,
            thread_id=payload.thread_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qa/resume")
async def qa_resume_post(payload: ResumeIn):
    try:
        return resume_question(
            thread_id=payload.thread_id,
            reply=payload.reply,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
