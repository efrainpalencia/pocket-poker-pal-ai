import os
from dotenv import load_dotenv
import uuid
from contextlib import asynccontextmanager
import uvicorn
import logging


from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from api.v1.routes import chat, chat_stream

from cli import cli_run
from graph.graph import graph

load_dotenv()


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("🚀 Application startup successful")
    yield
    logger.info("🛑 Application shutdown complete")

app = FastAPI(
    title="Pocket Poker Pal API",
    version="1.0",
    description="An API to access an AI-Powered Q&A poker assistant.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health Check / Root Route
@app.get("/")
def root(request: Request):
    message = "API running!"
    return {"message": message}


app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(chat_stream.router,
                   prefix="/api/v1/chat-stream", tags=["Chat-Stream"])

if __name__ == "__main__":
    # uvicorn.run("main:app", port=8000, reload=True)
    cli_run()
    # graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
