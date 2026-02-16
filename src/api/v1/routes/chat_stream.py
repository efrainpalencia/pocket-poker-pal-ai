from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse

from services.chat_stream_service import stream_qa, stream_resume

router = APIRouter()


@router.get("/qa/stream")
async def qa_stream(
    question: str = Query(..., min_length=1),
    thread_id: str | None = Query(None),
    request: Request = None,
    response: Response = None,
):
    """Stream Server-Sent Events (SSE) for a QA request.

    Query parameters:
      - question: required question text
      - thread_id: optional thread identifier to resume
    """

    try:
        return StreamingResponse(
            stream_qa(question=question, thread_id=thread_id),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qa/resume/stream")
async def qa_resume_stream(
    thread_id: str = Query(..., min_length=1),
    reply: str = Query(..., min_length=1),
    request: Request = None,
    response: Response = None,
):
    """Stream SSE while resuming an interrupted thread.

    Query parameters:
      - thread_id: required thread identifier
      - reply: required reply text to continue the thread
    """

    try:
        return StreamingResponse(
            stream_resume(thread_id=thread_id, reply=reply),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
