import logging
import os
import uuid
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.v1.routes import chat, chat_stream
from api.core.rate_limit import build_limiter
from cli import cli_run
from graph.graph import graph

load_dotenv()


logger = logging.getLogger(__name__)

limiter = build_limiter()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """FastAPI lifespan context manager.

    Logs a startup and shutdown message. Extendable for resource
    initialization/cleanup if needed.
    """

    logger.info("🚀 Application startup successful")
    yield
    logger.info("🛑 Application shutdown complete")


app = FastAPI(
    title="Pocket Poker Pal API",
    version="1.0",
    description="An API to access an AI-Powered Q&A poker assistant.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health Check / Root Route
@app.get("/health")
def health_check(request: Request):
    """Health-check / root endpoint.

    Returns a small JSON payload indicating service availability.
    """

    message = "healthy"
    return {"status": message}


app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(
    chat_stream.router, prefix="/api/v1/chat-stream", tags=["Chat-Stream"]
)

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
    # cli_run()
    # graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
