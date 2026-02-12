import pytest
from api.v1.routes.tests.conftest import (
    skip_if_missing_env,
    new_thread_id,
    collect_until_terminal_strict,
    assert_sse_terminal_contract,
)

pytestmark = pytest.mark.integration


@skip_if_missing_env
def test_sse_qa_stream_completes_or_interrupts(client, base_prefix):
    thread_id = new_thread_id("pytest-sse-qa")
    url = f"{base_prefix}/qa/stream" if base_prefix else "/qa/stream"

    with client.stream(
        "GET",
        url,
        params={
            "question": "In a tournament, what is a chip race and when is it performed?",
            "thread_id": thread_id,
        },
    ) as resp:
        assert resp.status_code == 200
        _, terminal = collect_until_terminal_strict(resp, max_events=60)

    assert_sse_terminal_contract(terminal, thread_id)


@skip_if_missing_env
def test_sse_resume_stream_flow(client, base_prefix):
    thread_id = new_thread_id("pytest-sse-resume")

    url_stream = f"{base_prefix}/qa/stream" if base_prefix else "/qa/stream"
    url_resume = f"{base_prefix}/qa/resume/stream" if base_prefix else "/qa/resume/stream"

    with client.stream(
        "GET",
        url_stream,
        params={"question": "Can I take this seat and what do I have to post?",
                "thread_id": thread_id},
    ) as resp1:
        assert resp1.status_code == 200
        _, terminal1 = collect_until_terminal_strict(resp1, max_events=60)

    assert terminal1 is not None

    if terminal1["type"] == "complete":
        assert_sse_terminal_contract(terminal1, thread_id)
        return

    prompt = terminal1.get("prompt") or {}
    ptype = prompt.get("type")

    reply = "cash-game" if ptype == "choose_ruleset" else (
        "Cash-game. New player is taking a seat and asks what blinds/post they must pay to enter."
    )

    with client.stream(
        "GET",
        url_resume,
        params={"thread_id": thread_id, "reply": reply},
    ) as resp2:
        assert resp2.status_code == 200
        _, terminal2 = collect_until_terminal_strict(resp2, max_events=120)

    assert terminal2 is not None
    assert terminal2.get("type") in {
        "complete", "needs_clarification", "error"}
    assert terminal2.get("thread_id") == thread_id

    # If it completes, enforce full contract.
    if terminal2.get("type") == "complete":
        assert_sse_terminal_contract(terminal2, thread_id)


@skip_if_missing_env
def test_sse_resume_invalid_reply_returns_error_or_needs_clarification(client, base_prefix):
    """
    If we reach a choose_ruleset interrupt, invalid reply should produce:
      - error OR
      - needs_clarification again
    """
    thread_id = new_thread_id("pytest-sse-invalid-reply")

    url_stream = f"{base_prefix}/qa/stream" if base_prefix else "/qa/stream"
    url_resume = f"{base_prefix}/qa/resume/stream" if base_prefix else "/qa/resume/stream"

    with client.stream(
        "GET",
        url_stream,
        params={
            "question": "If there is a dead small blind, can a new player be seated there and assume it?",
            "thread_id": thread_id,
        },
    ) as resp1:
        assert resp1.status_code == 200
        _, terminal1 = collect_until_terminal_strict(resp1, max_events=80)

    if terminal1.get("type") == "complete":
        return

    prompt1 = terminal1.get("prompt") or {}
    if prompt1.get("type") != "choose_ruleset":
        return

    with client.stream(
        "GET",
        url_resume,
        params={"thread_id": thread_id, "reply": "banana"},
    ) as resp2:
        assert resp2.status_code == 200
        _, terminal2 = collect_until_terminal_strict(resp2, max_events=60)

    assert terminal2.get("thread_id") == thread_id
    assert terminal2.get("type") in {"error", "needs_clarification"}

    if terminal2["type"] == "needs_clarification":
        p2 = (terminal2.get("prompt") or {}).get("type")
        assert p2 in {"choose_ruleset", "free_text"}


@skip_if_missing_env
def test_sse_double_interrupt_ruleset_then_free_text(client, base_prefix):
    """
    Best-effort double-interrupt path:
      choose_ruleset -> free_text -> (complete | needs_clarification | error)
    Tolerant to early completion.
    """
    thread_id = new_thread_id("pytest-sse-double")

    url_stream = f"{base_prefix}/qa/stream" if base_prefix else "/qa/stream"
    url_resume = f"{base_prefix}/qa/resume/stream" if base_prefix else "/qa/resume/stream"

    with client.stream(
        "GET",
        url_stream,
        params={"question": "Can I take this seat and what do I have to post?",
                "thread_id": thread_id},
    ) as resp1:
        assert resp1.status_code == 200
        _, terminal1 = collect_until_terminal_strict(resp1, max_events=80)

    if terminal1.get("type") == "complete":
        return

    p1 = (terminal1.get("prompt") or {}).get("type")
    if p1 != "choose_ruleset":
        return

    with client.stream(
        "GET",
        url_resume,
        params={"thread_id": thread_id, "reply": "cash-game"},
    ) as resp2:
        assert resp2.status_code == 200
        _, terminal2 = collect_until_terminal_strict(resp2, max_events=120)

    if terminal2.get("type") == "complete":
        assert_sse_terminal_contract(terminal2, thread_id)
        return

    # Expect (or at least allow) free_text as the next clarification
    if terminal2.get("type") != "needs_clarification":
        return

    p2 = (terminal2.get("prompt") or {}).get("type")
    if p2 != "free_text":
        return

    with client.stream(
        "GET",
        url_resume,
        params={
            "thread_id": thread_id,
            "reply": "Cash-game context. New player sat mid-orbit and asks what blinds/posts are required to enter.",
        },
    ) as resp3:
        assert resp3.status_code == 200
        _, terminal3 = collect_until_terminal_strict(resp3, max_events=160)

    assert terminal3.get("type") in {
        "complete", "needs_clarification", "error"}
    assert terminal3.get("thread_id") == thread_id
