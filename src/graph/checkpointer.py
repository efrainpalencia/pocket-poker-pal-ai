import os

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Hold references to any entered context-managers so they are not
# garbage-collected and closed unexpectedly for the lifetime of the
# process (useful when 3rd-party factories return context-managers).
_active_context_managers: list = []


def build_checkpointer():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        # Local dev fallback
        return MemorySaver()

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ModuleNotFoundError:
        # Postgres saver implementation not available in installed packages.
        # Fall back to in-memory saver so the app can start without this
        # optional dependency (useful for local/dev environments).
        return MemorySaver()

    # Some implementations return a context-manager (e.g. a
    # @contextlib.contextmanager) from `from_conn_string`. In that case
    # the returned value won't have `setup()` directly. Detect and
    # handle both cases: either call `setup()` on the saver directly,
    # or enter the context manager and call `setup()` on the entered
    # object.
    saver = PostgresSaver.from_conn_string(database_url)

    # Try to support both plain saver objects and context-manager factories.
    try:
        if hasattr(saver, "__enter__") and hasattr(saver, "__exit__"):
            # Enter and keep the context manager alive for the process
            # lifetime by storing it in `_active_context_managers`.
            real_saver = saver.__enter__()
            try:
                real_saver.setup()
            except Exception:
                # Cleanup and fall back to MemorySaver.
                try:
                    saver.__exit__(None, None, None)
                except Exception:
                    pass
                return MemorySaver()

            _active_context_managers.append(saver)
            return real_saver

        # Plain saver object
        saver.setup()
        return saver
    except Exception:
        # If anything goes wrong (e.g. cannot connect to DB), fall back
        # to an in-memory saver so tests and local runs continue to work.
        return MemorySaver()
