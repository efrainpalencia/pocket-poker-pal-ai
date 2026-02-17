import os
from contextlib import ExitStack
from dataclasses import dataclass

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


@dataclass
class CheckpointerHandle:
    saver: object
    stack: ExitStack | None = None


def close_checkpointer(handle: CheckpointerHandle | None) -> None:
    """Close resources associated with this checkpointer handle."""
    if not handle:
        return
    if handle.stack is not None:
        handle.stack.close()


def build_checkpointer() -> CheckpointerHandle:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return CheckpointerHandle(saver=MemorySaver(), stack=None)

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ModuleNotFoundError:
        return CheckpointerHandle(saver=MemorySaver(), stack=None)

    saver_or_cm = PostgresSaver.from_conn_string(database_url)

    # If PostgresSaver returns an ASYNC context manager, handle in lifespan
    # with AsyncExitStack (requires build_checkpointer to become async).
    if hasattr(saver_or_cm, "__aenter__") and hasattr(saver_or_cm, "__aexit__"):
        raise RuntimeError(
            "PostgresSaver.from_conn_string returned an async context manager. "
            "Use an AsyncExitStack in lifespan and await __aenter__()."
        )

    # Context manager case: keep it alive for this app lifetime
    if hasattr(saver_or_cm, "__enter__") and hasattr(saver_or_cm, "__exit__"):
        stack = ExitStack()
        saver = stack.enter_context(saver_or_cm)
        saver.setup()
        return CheckpointerHandle(saver=saver, stack=stack)

    # Plain saver object
    saver_or_cm.setup()
    return CheckpointerHandle(saver=saver_or_cm, stack=None)
