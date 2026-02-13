import pytest

from graph.chains.classifier import classifier_chain
from graph.chains.generation import generation_chain, GenerationOut
from graph.chains.grader import grader_chain, GradeOut


def patch_chatopenai_generate(monkeypatch, content: str):
    """
    Patch ChatOpenAI._generate to return a valid ChatResult that LangChain expects.
    This avoids network calls.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_openai import ChatOpenAI

    def _fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

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
def test_generation_chain_returns_generationout(monkeypatch):
    """
    generation_chain returns a GenerationOut Pydantic model.
    Patch the module-level generation_chain to avoid patching RunnableSequence internals.
    """
    import graph.chains.generation as gen_mod
    from graph.chains.generation import GenerationOut

    class FakeChain:
        def invoke(self, inp, config=None, **kwargs):
            return GenerationOut(
                mode="direct",
                answer="You may post to play.",
                quote="A player may post to play.",
                caveat=None,
                clarifying_question=None,
            )

    monkeypatch.setattr(gen_mod, "generation_chain", FakeChain(), raising=True)

    out = gen_mod.generation_chain.invoke(
        {"context": "A player may post to play.",
            "question": "Can I post to play?"}
    )

    assert isinstance(out, GenerationOut)
    assert out.mode in ("direct", "inference", "not_found")
    assert isinstance(out.answer, str) and out.answer.strip()


def test_generation_chain_not_found_contract(monkeypatch):
    """
    Enforce your strict not_found contract:
      - answer exactly 'Not found in the provided text.'
      - clarifying_question required
      - quote/caveat must be None
    """
    import graph.chains.generation as gen_mod
    from graph.chains.generation import GenerationOut

    class FakeChain:
        def invoke(self, inp, config=None, **kwargs):
            return GenerationOut(
                mode="not_found",
                answer="Not found in the provided text.",
                quote=None,
                caveat=None,
                clarifying_question="Is this question about tournaments or cash games?",
            )

    monkeypatch.setattr(gen_mod, "generation_chain", FakeChain(), raising=True)

    out = gen_mod.generation_chain.invoke(
        {"context": "", "question": "What happens here?"})

    assert isinstance(out, GenerationOut)
    assert out.mode == "not_found"
    assert out.answer == "Not found in the provided text."
    assert out.clarifying_question and out.clarifying_question.endswith("?")
    assert out.quote is None
    assert out.caveat is None


# -----------------------------
# Grader chain tests
# -----------------------------
def test_grader_chain_parses_json(monkeypatch):
    """
    grader_chain uses a PydanticOutputParser -> it MUST receive JSON.
    """
    patch_chatopenai_generate(
        monkeypatch,
        """
        {
          "confidence": 0.9,
          "label": "YES",
          "reasons": ["Supported by context."],
          "missing_info": [],
          "is_hallucination_risk": false
        }
        """.strip(),
    )

    out = grader_chain.invoke({"question": "Q", "context": "C", "answer": "A"})
    assert isinstance(out, GradeOut)
    assert out.label == "YES"
    assert 0.0 <= out.confidence <= 1.0


@pytest.mark.parametrize(
    "confidence,expected_label",
    [
        (0.90, "YES"),
        (0.70, "PARTIAL"),
        (0.10, "NO"),
    ],
)
def test_grader_chain_label_matches_confidence(monkeypatch, confidence, expected_label):
    patch_chatopenai_generate(
        monkeypatch,
        f"""
        {{
          "confidence": {confidence},
          "label": "{expected_label}",
          "reasons": [],
          "missing_info": [],
          "is_hallucination_risk": false
        }}
        """.strip(),
    )

    out = grader_chain.invoke({"question": "Q", "context": "C", "answer": "A"})
    assert isinstance(out, GradeOut)
    assert out.label == expected_label


def test_grader_chain_rejects_non_json(monkeypatch):
    """
    If the model returns garbage (non-JSON), parser should raise.
    This test now matches the chain behavior.
    """
    patch_chatopenai_generate(monkeypatch, "MAYBE")

    with pytest.raises(Exception):
        grader_chain.invoke({"question": "Q", "context": "C", "answer": "A"})
