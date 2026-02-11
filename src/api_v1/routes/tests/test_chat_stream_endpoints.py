import json
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from main import app


BASE = "/api-v1/chat-stream"


def _read_sse_events(resp) -> List[Dict[str, Any]]:
    """
    Parse SSE response lines from FastAPI TestClient streaming response.
    We only care about lines like:
      data: {...json...}
    """
    events: List[Dict[str, Any]] = []
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue

        line = raw_line.decode(
            "utf-8") if isinstance(raw_line, (bytes, bytearray)) else raw_line
        if not line.startswith("data:"):
            continue

        payload = line[len("data:"):].strip()
        if not payload:
            continue

        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            # If anything non-JSON gets emitted, ignore it
            continue

    return events


def _find_event(events: List[Dict[str, Any]], event_type: str) -> Optional[Dict[str, Any]]:
    for e in events:
        if e.get("type") == event_type:
            return e
    return None


def _collect_text(events: List[Dict[str, Any]], event_type: str, field: str) -> str:
    parts = []
    for e in events:
        if e.get("type") == event_type and e.get(field):
            parts.append(e[field])
    return "".join(parts)


@pytest.mark.integration
def test_qa_stream_then_resume_stream_tournament():
    client = TestClient(app)

    question = "If there is a dead small blind, can a new player be sat in that position an assume the small blind?"

    # 1) Start stream
    with client.stream(
        "GET",
        f"{BASE}/qa/stream",
        params={"question": question},
        headers={"Accept": "text/event-stream"},
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = _read_sse_events(resp)

    start = _find_event(events, "start")
    assert start is not None, f"No start event found. Events: {events}"
    thread_id = start.get("thread_id")
    assert thread_id, "start event missing thread_id"

    # Depending on your route_or_clarify heuristic/classifier, this may or may not interrupt.
    needs = _find_event(events, "needs_clarification")
    complete = _find_event(events, "complete")

    assert needs or complete, f"Expected needs_clarification or complete. Events: {events}"

    # If it completed immediately (no clarification needed), validate generation and exit.
    if complete:
        assert complete.get(
            "generation") is not None, f"Complete generation was null. Event: {complete}"
        assert isinstance(complete.get("generation"), str)
        assert complete["generation"].strip() != ""
        return

    # 2) Resume stream (clarify tournament)
    with client.stream(
        "GET",
        f"{BASE}/qa/resume/stream",
        params={"thread_id": thread_id, "reply": "tournament"},
        headers={"Accept": "text/event-stream"},
    ) as resp2:
        assert resp2.status_code == 200
        assert "text/event-stream" in resp2.headers.get("content-type", "")

        resume_events = _read_sse_events(resp2)

    resume = _find_event(resume_events, "resume")
    assert resume is not None, f"No resume event found. Events: {resume_events}"

    # You should see some token events (buffered but present)
    token_text = _collect_text(resume_events, "token", "text")
    assert token_text.strip() != "", "No token text collected from resume stream."

    # After resume, the graph may either:
    # 1) complete with an answer
    # 2) ask for more details (retry_or_clarify)
    complete2 = _find_event(resume_events, "complete")
    needs2 = _find_event(resume_events, "needs_clarification")

    assert complete2 or needs2, (
        f"Expected complete or needs_clarification after resume. Events: {resume_events}"
    )

    if complete2:
        assert complete2.get(
            "generation") is not None, f"Complete generation was null. Event: {complete2}"
        assert isinstance(complete2.get("generation"), str)
        assert complete2["generation"].strip() != ""
    else:
        # If it needs more clarification, ensure we got a prompt
        assert needs2.get(
            "prompt"), f"needs_clarification missing prompt. Event: {needs2}"


@pytest.mark.integration
def test_qa_stream_then_resume_stream_cash_game():
    client = TestClient(app)

    question = "If there is a dead small blind, can a new player be sat in that position an assume the small blind?"

    # 1) Start stream
    with client.stream(
        "GET",
        f"{BASE}/qa/stream",
        params={"question": question},
        headers={"Accept": "text/event-stream"},
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = _read_sse_events(resp)

    start = _find_event(events, "start")
    assert start is not None, f"No start event found. Events: {events}"
    thread_id = start.get("thread_id")
    assert thread_id, "start event missing thread_id"

    # Depending on your route_or_clarify heuristic/classifier, this may or may not interrupt.
    needs = _find_event(events, "needs_clarification")
    complete = _find_event(events, "complete")

    assert needs or complete, f"Expected needs_clarification or complete. Events: {events}"

    # If it completed immediately (no clarification needed), validate generation and exit.
    if complete:
        assert complete.get(
            "generation") is not None, f"Complete generation was null. Event: {complete}"
        assert isinstance(complete.get("generation"), str)
        assert complete["generation"].strip() != ""
        return

    # 2) Resume stream (clarify tournament)
    with client.stream(
        "GET",
        f"{BASE}/qa/resume/stream",
        params={"thread_id": thread_id, "reply": "cash-game"},
        headers={"Accept": "text/event-stream"},
    ) as resp2:
        assert resp2.status_code == 200
        assert "text/event-stream" in resp2.headers.get("content-type", "")

        resume_events = _read_sse_events(resp2)

    resume = _find_event(resume_events, "resume")
    assert resume is not None, f"No resume event found. Events: {resume_events}"

    # You should see some token events (buffered but present)
    token_text = _collect_text(resume_events, "token", "text")
    assert token_text.strip() != "", "No token text collected from resume stream."

    # After resume, the graph may either:
    # 1) complete with an answer
    # 2) ask for more details (retry_or_clarify)
    complete2 = _find_event(resume_events, "complete")
    needs2 = _find_event(resume_events, "needs_clarification")

    assert complete2 or needs2, (
        f"Expected complete or needs_clarification after resume. Events: {resume_events}"
    )

    if complete2:
        assert complete2.get(
            "generation") is not None, f"Complete generation was null. Event: {complete2}"
        assert isinstance(complete2.get("generation"), str)
        assert complete2["generation"].strip() != ""
    else:
        # If it needs more clarification, ensure we got a prompt
        assert needs2.get(
            "prompt"), f"needs_clarification missing prompt. Event: {needs2}"
