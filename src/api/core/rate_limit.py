import os
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _client_ip(request: Request) -> str:
    """
    Prefer X-Forwarded-For if present (behind a proxy), else fallback.
    NOTE: only trust XFF if your proxy is configured correctly.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        # take first IP in the list
        return xff.split(",")[0].strip()
    return get_remote_address(request)


def build_limiter() -> Limiter:
    # e.g. "redis://localhost:6379/0"
    storage_uri = os.getenv("REDIS_URL")
    headers_enabled = os.getenv(
        "RATE_LIMIT_HEADERS_ENABLED", "false").lower() == "false"

    if storage_uri:
        return Limiter(
            key_func=_client_ip,
            storage_uri=storage_uri,
            headers_enabled=headers_enabled,
            # Optional global defaults (applies when no per-route limit is set)
            default_limits=[os.getenv("RATE_LIMIT_DEFAULT", "5/minute")],
        )
    # Dev fallback (in-memory; not good for multi-worker prod)
    return Limiter(
        key_func=_client_ip,
        headers_enabled=headers_enabled,
        default_limits=[os.getenv("RATE_LIMIT_DEFAULT", "5/minute")],
    )


limiter = build_limiter()
