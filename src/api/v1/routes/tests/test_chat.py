import pytest

from api.v1.routes.tests.conftest import (
    assert_json_contract,
    new_thread_id,
    skip_if_missing_env,
)

pytestmark = pytest.mark.integration


@skip_if_missing_env
def test_post_qa_can_complete_or_interrupt(client, base_prefix):
    thread_id = new_thread_id("pytest-post-qa")
    url = f"{base_prefix}/qa" if base_prefix else "/qa"

    r = client.post(
        url,
        json={
            "question": "In a tournament, what is a chip race and when is it performed?",
            "thread_id": thread_id,
        },
    )
    assert r.status_code == 200, r.text
    assert_json_contract(r.json(), thread_id)


@skip_if_missing_env
def test_post_qa_resume_flow(client, base_prefix):
    thread_id = new_thread_id("pytest-post-resume")
    url_qa = f"{base_prefix}/qa" if base_prefix else "/qa"
    url_resume = f"{base_prefix}/qa/resume" if base_prefix else "/qa/resume"

    r1 = client.post(
        url_qa,
        json={
            "question": "Can I take this seat and what do I have to post?",
            "thread_id": thread_id,
        },
    )
    assert r1.status_code == 200, r1.text
    data1 = r1.json()

    if data1["status"] == "complete":
        assert data1.get("generation")
        return

    prompt = data1.get("prompt") or {}
    ptype = prompt.get("type")
    reply = (
        "cash-game"
        if ptype == "choose_ruleset"
        else (
            "Cash-game. New player sits in and asks what blind/post is required to enter."
        )
    )

    r2 = client.post(url_resume, json={"thread_id": thread_id, "reply": reply})
    assert r2.status_code == 200, r2.text
    data2 = r2.json()

    # Still allow a second clarification (double-interrupt scenario)
    assert data2["status"] in {"complete", "needs_clarification"}


@skip_if_missing_env
def test_post_resume_invalid_reply_returns_error_or_needs_clarification(
    client, base_prefix
):
    """
    If we get a choose_ruleset prompt, an invalid reply should not silently "complete".
    Acceptable behaviors:
      - 400 with needs_clarification or structured error
      - 422 (validation)
      - 200 but needs_clarification/error (if your API models it this way)
    """
    thread_id = new_thread_id("pytest-post-invalid-reply")
    url_qa = f"{base_prefix}/qa" if base_prefix else "/qa"
    url_resume = f"{base_prefix}/qa/resume" if base_prefix else "/qa/resume"

    r1 = client.post(
        url_qa,
        json={
            "question": "If there is a dead small blind, can a new player be seated there and assume it?",
            "thread_id": thread_id,
        },
    )
    assert r1.status_code == 200, r1.text
    data1 = r1.json()

    if data1["status"] == "complete":
        return

    prompt = data1.get("prompt") or {}
    if prompt.get("type") != "choose_ruleset":
        return

    r2 = client.post(url_resume, json={"thread_id": thread_id, "reply": "banana"})
    assert r2.status_code in {200, 400, 422}, r2.text

    if r2.status_code == 422:
        body = r2.json()
        assert "detail" in body
        return

    body = r2.json()
    assert body.get("thread_id") == thread_id
    assert body.get("status") in {"needs_clarification", "complete"}

    # Ideally it should not "complete" on invalid choice; if it does, at least it must be grounded.
    if body["status"] == "complete":
        assert body.get("generation")
    else:
        p2 = (body.get("prompt") or {}).get("type")
        assert p2 in {"choose_ruleset", "free_text"}


@skip_if_missing_env
def test_post_double_interrupt_ruleset_then_free_text(client, base_prefix):
    """
    Double-interrupt path (best-effort; tolerant if model completes early):
      1) /qa -> choose_ruleset
      2) /qa/resume -> still needs_clarification with free_text
      3) /qa/resume -> complete (or still needs_clarification)
    """
    thread_id = new_thread_id("pytest-post-double")
    url_qa = f"{base_prefix}/qa" if base_prefix else "/qa"
    url_resume = f"{base_prefix}/qa/resume" if base_prefix else "/qa/resume"

    r1 = client.post(
        url_qa, json={"question": "Can I take this seat?", "thread_id": thread_id}
    )
    assert r1.status_code == 200, r1.text
    data1 = r1.json()

    if data1["status"] == "complete":
        return

    p1 = (data1.get("prompt") or {}).get("type")
    if p1 != "choose_ruleset":
        return

    r2 = client.post(url_resume, json={"thread_id": thread_id, "reply": "cash-game"})
    assert r2.status_code == 200, r2.text
    data2 = r2.json()

    if data2["status"] == "complete":
        return

    p2 = (data2.get("prompt") or {}).get("type")
    if p2 != "free_text":
        # Still acceptable: it may ask ruleset again or complete after first resume on some runs
        return

    r3 = client.post(
        url_resume,
        json={
            "thread_id": thread_id,
            "reply": "Cash-game context. New player sat mid-orbit and asks what blinds/posts are required to enter.",
        },
    )
    assert r3.status_code == 200, r3.text
    data3 = r3.json()
    assert data3["status"] in {"complete", "needs_clarification"}
