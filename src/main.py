import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.core.rate_limit import build_limiter
from api.v1.routes import chat, chat_stream
from graph.checkpointer import build_checkpointer, close_checkpointer
from graph.graph import build_graph
from graph.workflow import write_graph_png

load_dotenv()

logger = logging.getLogger(__name__)
limiter = build_limiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Initializing LangGraph resources...")

    handle = build_checkpointer()
    graph = build_graph(handle.saver)

    app.state.checkpointer_handle = handle
    app.state.graph = graph

    logger.info("✅ LangGraph initialized")
    try:
        yield
    finally:
        logger.info("🛑 Closing checkpointer...")
        close_checkpointer(app.state.checkpointer_handle)
        logger.info("✅ Shutdown complete")


app = FastAPI(
    title="Pocket Poker Pal API",
    version="1.0",
    description="An API to access an AI-Powered Q&A poker assistant.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check(request: Request):
    return {
        "status": "healthy",
        "graph_initialized": hasattr(request.app.state, "graph"),
    }


app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(
    chat_stream.router, prefix="/api/v1/chat-stream", tags=["Chat-Stream"]
)


# if __name__ == "__main__":
# uvicorn.run("main:app", port=8000, reload=True)
# cli_run()
# write_graph_png()
