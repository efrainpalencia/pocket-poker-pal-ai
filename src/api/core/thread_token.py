import base64
import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException

# ENV VARS (set these in Railway)
# THREAD_TOKEN_SECRET="a-long-random-secret"   (required)
# THREAD_TOKEN_TTL_SECONDS="7200"              (optional; default 2 hours)

_SECRET = os.getenv("THREAD_TOKEN_SECRET", "").encode("utf-8")
_TTL = int(os.getenv("THREAD_TOKEN_TTL_SECONDS", "7200"))

if not _SECRET:
    # In production you want this set; in dev you can set it in .env
    # We don't raise at import time to keep tests/dev flexible.
    pass


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def _sign(payload: bytes, secret: bytes) -> bytes:
    return hmac.new(secret, payload, hashlib.sha256).digest()


@dataclass(frozen=True)
class ThreadTokenClaims:
    thread_id: str
    issued_at: int
    expires_at: int
    ip_hint: Optional[str] = None  # optional binding


def create_thread_token(thread_id: str, ip: Optional[str] = None) -> str:
    """
    Create a signed thread token for a given thread_id.

    Token format (no JWT deps):
      base64url(payload) + "." + base64url(sig)

    payload (utf-8):
      v1|<thread_id>|<iat>|<exp>|<ip-or-empty>

    - ip is optional. If provided, the token is bound to that IP string.
    """
    if not _SECRET:
        raise RuntimeError("THREAD_TOKEN_SECRET is not set")

    now = int(time.time())
    exp = now + _TTL

    ip_part = ip or ""
    payload_str = f"v1|{thread_id}|{now}|{exp}|{ip_part}"
    payload = payload_str.encode("utf-8")

    sig = _sign(payload, _SECRET)

    return f"{_b64url_encode(payload)}.{_b64url_encode(sig)}"


def verify_thread_token(
    token: str,
    thread_id: str,
    ip: Optional[str] = None,
    *,
    leeway_seconds: int = 30,
) -> ThreadTokenClaims:
    """
    Verify a thread token. Raises HTTPException(401) if invalid.

    - thread_id must match
    - exp must be >= now (with small leeway)
    - if token contains an ip_hint, it must match provided ip
    """
    if not _SECRET:
        raise RuntimeError("THREAD_TOKEN_SECRET is not set")

    try:
        payload_b64, sig_b64 = token.split(".", 1)
        payload = _b64url_decode(payload_b64)
        sig = _b64url_decode(sig_b64)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid thread token format")

    expected_sig = _sign(payload, _SECRET)
    if not hmac.compare_digest(sig, expected_sig):
        raise HTTPException(status_code=401, detail="Invalid thread token signature")

    try:
        payload_str = payload.decode("utf-8")
        parts = payload_str.split("|")
        if len(parts) != 5:
            raise ValueError("bad parts")
        version, tok_thread_id, iat_s, exp_s, ip_hint = parts
        if version != "v1":
            raise ValueError("bad version")

        iat = int(iat_s)
        exp = int(exp_s)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid thread token payload")

    if tok_thread_id != thread_id:
        raise HTTPException(
            status_code=401, detail="Thread token does not match thread"
        )

    now = int(time.time())
    if exp + leeway_seconds < now:
        raise HTTPException(status_code=401, detail="Thread token expired")

    # Optional IP binding:
    # - If token includes an ip_hint, require match.
    # - If token has no ip_hint, allow any ip.
    if ip_hint and ip and ip_hint != ip:
        raise HTTPException(status_code=401, detail="Thread token IP mismatch")

    return ThreadTokenClaims(
        thread_id=tok_thread_id,
        issued_at=iat,
        expires_at=exp,
        ip_hint=ip_hint or None,
    )
