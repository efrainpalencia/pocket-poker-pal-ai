import os
import uuid

import pytest
from langgraph.types import Command

from graph.graph import graph

REQUIRED_ENV = ["OPENAI_API_KEY", "PINECONE_API_KEY", "INDEX_NAME"]


def _missing_env():
    return [k for k in REQUIRED_ENV if not os.getenv(k)]


pytestmark = pytest.mark.integration


def _pick_resume(interrupt_payload, defaults):
    """
    interrupt_payload is usually a list of Interrupt objects.
    We expect our structured interrupt to be in Interrupt.value (dict).
    defaults is a dict with keys: ruleset_choice, free_text
    """
    try:
        first = interrupt_payload[0]
        val = getattr(first, "value", None)
        if isinstance(val, dict):
            if val.get("type") == "choose_ruleset":
                return defaults["ruleset_choice"]
            if val.get("type") == "free_text":
                return defaults["free_text"]
    except Exception:
        pass

    # fallback if shape differs
    return defaults["free_text"]


def run_until_done(inputs, config, defaults, max_steps=20):
    """
    Drives the graph through interrupts until a final state is returned.
    defaults:
      - ruleset_choice: "tournament" | "cash-game"
      - free_text: clarification string
    """
    out = graph.invoke(inputs, config=config)

    for _ in range(max_steps):
        # If we're done, return immediately
        if not (isinstance(out, dict) and "__interrupt__" in out):
            return out

        # Pull your structured interrupt payload (we passed dicts to interrupt())
        intr = out["__interrupt__"][0]
        payload = getattr(intr, "value", None)

        if isinstance(payload, dict) and payload.get("type") == "choose_ruleset":
            reply = defaults["ruleset_choice"]
        else:
            reply = defaults["free_text"]

        # Resume and capture the result
        out = graph.invoke(Command(resume=reply), config=config)

    raise AssertionError("Graph did not complete within max_steps (possible loop).")


@pytest.mark.skipif(_missing_env(), reason=f"Missing env vars: {_missing_env()}")
def test_graph_end_to_end_tournament_success():
    config = {"configurable": {"thread_id": "pytest-e2e-tournament-1"}}

    out = run_until_done(
        inputs={
            "question": "In a tournament, what is a chip race and when is it performed?"
        },
        config=config,
        defaults={
            "ruleset_choice": "tournament",
            "free_text": "Tournament rules. Chip race procedure and when it happens.",
        },
    )

    assert (
        out.get("confidence", 0.0) >= 0.60
    ), f"Low confidence: {out.get('confidence')}"
    assert out.get("grounded") is True
    assert out.get("generation")


@pytest.mark.skipif(_missing_env(), reason=f"Missing env vars: {_missing_env()}")
def test_graph_end_to_end_interrupt_and_resume():
    config = {"configurable": {"thread_id": "pytest-e2e-interrupt-1"}}

    out = run_until_done(
        inputs={"question": "Is it a misdeal"},
        config=config,
        defaults={
            "ruleset_choice": "cash-game",
            "free_text": "Cash-game. The dealer dealt the wrong player first.",
        },
    )

    assert (
        out.get("confidence", 0.0) >= 0.60
    ), f"Low confidence after resume: {out.get('confidence')}"
    assert out.get("grounded") is True
    assert out.get("generation")


@pytest.mark.skipif(_missing_env(), reason=f"Missing env vars: {_missing_env()}")
def test_tournament_fallback_hits_section_j_when_needed():
    config = {"configurable": {"thread_id": "pytest-section-j-1"}}

    out = run_until_done(
        inputs={
            "question": "If there is a dead small blind, can a new player be seated there and assume the small blind?"
        },
        config=config,
        defaults={
            "ruleset_choice": "tournament",
            "free_text": "Tournament. Question is about being seated into a dead small blind position.",
        },
    )

    assert out.get("generation")
    assert out.get("grounded") is True

    docs = out.get("documents", []) or []
    hit_section_j = any((d.metadata or {}).get("section") == "Section_J" for d in docs)

    # ✅ Only require Section J if fallback condition was met
    if float(out.get("retrieval_strength", 1.0)) < 0.35:
        assert hit_section_j, "Expected Section_J when primary retrieval is weak"
