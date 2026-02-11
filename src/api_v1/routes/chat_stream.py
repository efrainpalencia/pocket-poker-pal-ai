from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from services.chat_stream_service import stream_qa, stream_resume

router = APIRouter()


@router.get("/qa/stream")
async def qa_stream(
    question: str = Query(..., min_length=1),
    thread_id: str | None = Query(None),
):
    try:
        return StreamingResponse(stream_qa(question=question, thread_id=thread_id),
                                 media_type="text/event-stream")
    except Exception:
        raise HTTPException(
            status_code=500, detail="Error occurred while streaming QA.")


@router.get("/qa/resume/stream")
async def qa_resume_stream(
    thread_id: str = Query(..., min_length=1),
    reply: str = Query(..., min_length=1),
):
    try:
        return StreamingResponse(stream_resume(thread_id=thread_id, reply=reply),
                                 media_type="text/event-stream")
    except Exception:
        raise HTTPException(
            status_code=500, detail="Error occurred while streaming QA resume.")
