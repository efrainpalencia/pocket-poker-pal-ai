import pytest

from graph.chains.classifier import classifier_chain
from graph.chains.generation import generation_chain
from graph.chains.grader import grader_chain


def patch_chatopenai_generate(monkeypatch, content: str):
    """
    Patch ChatOpenAI._generate to return a valid ChatResult that LangChain expects.
    This avoids network calls and avoids pydantic Generation validation errors.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    def _fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    monkeypatch.setattr(ChatOpenAI, "_generate", _fake_generate, raising=True)


# -----------------------------
# Classifier chain tests
# -----------------------------
def test_classifier_chain_returns_expected_label(monkeypatch):
    patch_chatopenai_generate(monkeypatch, "tournament")

    out = classifier_chain.invoke(
        {"question": "In a tournament, can I late reg?"})
    assert isinstance(out, str)
    assert out.strip().lower() in ("tournament", "cash-game", "unknown")


def test_classifier_chain_unknown(monkeypatch):
    patch_chatopenai_generate(monkeypatch, "unknown")

    out = classifier_chain.invoke({"question": "Can I do the thing?"})
    assert out.strip().lower() == "unknown"


# -----------------------------
# Generation chain tests
# -----------------------------
def test_generation_chain_returns_string(monkeypatch):
    patch_chatopenai_generate(monkeypatch, "Yes. You may post to play.")

    out = generation_chain.invoke(
        {"context": "Rule text...", "question": "Can I post to play?"})
    assert isinstance(out, str)
    assert out.strip() != ""


# -----------------------------
# Grader chain tests
# -----------------------------
@pytest.mark.parametrize("verdict,expected", [
    ("YES", True),
    ("Yes", True),
    (" NO ", False),
    ("no", False),
])
def test_grader_chain_basic_yes_no(monkeypatch, verdict, expected):
    patch_chatopenai_generate(monkeypatch, verdict)

    out = grader_chain.invoke({"question": "Q", "context": "C", "answer": "A"})
    grounded = out.strip().upper().startswith("Y")
    assert grounded is expected


def test_grader_chain_rejects_garbage(monkeypatch):
    patch_chatopenai_generate(monkeypatch, "MAYBE")

    out = grader_chain.invoke({"question": "Q", "context": "C", "answer": "A"})
    grounded = out.strip().upper().startswith("Y")
    assert grounded is False
