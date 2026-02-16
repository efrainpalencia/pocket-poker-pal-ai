from fastapi import APIRouter, HTTPException, Request, Response

from api.core.rate_limit import limiter
from schemas.chat_schema import AskIn, ResumeIn
from services.chat_service import ask_question, resume_question

router = APIRouter()


@router.post("/qa")
@limiter.limit("5/minute")
async def qa_post(payload: AskIn, request: Request, response: Response):
    """HTTP POST handler for non-streaming question answering.

    Accepts an `AskIn` payload and returns the result from
    `services.chat_service.ask_question`. Converts exceptions into
    HTTP 500 responses.
    """

    try:
        return ask_question(
            question=payload.question,
            thread_id=payload.thread_id,
            request=request,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qa/resume")
@limiter.limit("5/minute")
async def qa_resume_post(payload: ResumeIn, request: Request, response: Response):
    """HTTP POST handler to resume an interrupted thread.

    Accepts `ResumeIn` and forwards to `services.chat_service.resume_question`.
    """

    try:
        return resume_question(
            thread_id=payload.thread_id,
            thread_token=payload.thread_token,
            reply=payload.reply,
            request=request,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
