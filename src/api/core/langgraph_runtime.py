from __future__ import annotations

import logging
from typing import Any

from fastapi import Request

from graph.checkpointer import (CheckpointerHandle, build_checkpointer,
                                close_checkpointer)
from graph.graph import build_graph

logger = logging.getLogger(__name__)


def ensure_graph(request: Request) -> Any:
    graph = getattr(request.app.state, "graph", None)
    handle = getattr(request.app.state, "checkpointer_handle", None)
    if graph is None or handle is None:
        raise RuntimeError(
            "LangGraph not initialized. Ensure lifespan sets app.state.graph"
        )
    return graph


def rebuild_graph(request: Request) -> Any:
    old: CheckpointerHandle | None = getattr(
        request.app.state, "checkpointer_handle", None
    )

    try:
        close_checkpointer(old)
    except Exception:
        logger.exception("Failed to close old checkpointer handle")

    new_handle = build_checkpointer()
    new_graph = build_graph(new_handle.saver)

    request.app.state.checkpointer_handle = new_handle
    request.app.state.graph = new_graph

    logger.warning("♻️ Rebuilt LangGraph checkpointer + graph after disconnect")
    return new_graph
