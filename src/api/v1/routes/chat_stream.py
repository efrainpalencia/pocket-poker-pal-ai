from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from services.chat_stream_service import stream_qa, stream_resume

router = APIRouter()


@router.get("/qa/stream")
async def qa_stream(
    request: Request,
    question: str = Query(..., min_length=1),
    thread_id: str | None = Query(None),
):
    """
    Stream Server-Sent Events (SSE) for a QA request.

    Query parameters:
      - question: required question text
      - thread_id: optional thread identifier to resume
    """

    try:
        return StreamingResponse(
            stream_qa(
                question=question,
                thread_id=thread_id,
                request=request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # prevents nginx buffering
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qa/resume/stream")
async def qa_resume_stream(
    request: Request,
    thread_id: str = Query(..., min_length=1),
    thread_token: str = Query(..., min_length=10),
    reply: str = Query(..., min_length=1),
):
    """
    Stream SSE while resuming an interrupted thread.

    Query parameters:
      - thread_id: required thread identifier
      - thread_token: required thread security token
      - reply: required reply text to continue the thread
    """

    try:
        return StreamingResponse(
            stream_resume(
                thread_id=thread_id,
                thread_token=thread_token,
                reply=reply,
                request=request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
