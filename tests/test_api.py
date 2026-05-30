"""End-to-end tests for /ask plus core guardrail/gate behaviour.

Builds the real pipeline (embeddings + FAISS index) once via the FastAPI
lifespan. Tests that need answer generation are skipped when no LLM key is
configured (refusal/guardrail paths still run fully offline).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.guardrails import check_high_risk
from app.main import app

needs_llm = pytest.mark.skipif(
    not settings.use_llm, reason="requires an LLM API key (see .env.example)"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _ask(client, question: str) -> dict:
    resp = client.post("/ask", json={"question": question})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_response_schema(client):
    # Use a refusal path so this runs without an LLM key.
    data = _ask(client, "What is the capital of France?")
    required = {"answer", "evidence_used", "evidence_sufficient", "guardrail_triggered"}
    assert required.issubset(set(data))
    assert "groundedness" in data  # additive transparency field


@needs_llm
def test_groundedness_present_on_answer(client):
    data = _ask(client, "What is HFpEF and what are its common symptoms?")
    assert data["groundedness"] is not None
    assert 0.0 <= data["groundedness"]["score"] <= 1.0


@needs_llm
def test_general_education_is_answered_with_citations(client):
    data = _ask(client, "What is HFpEF and what are its common symptoms?")
    assert data["evidence_sufficient"] is True
    assert data["guardrail_triggered"] is False
    assert len(data["evidence_used"]) > 0
    for item in data["evidence_used"]:
        assert item["document_id"].startswith("doc_")
        assert item["chunk_id"].startswith("chunk_")
        assert 0.0 <= item["similarity_score"] <= 1.0


def test_insufficient_evidence_is_refused(client):
    data = _ask(client, "What is the recommended treatment for a broken ankle?")
    assert data["evidence_sufficient"] is False
    assert data["guardrail_triggered"] is False
    assert "enough" in data["answer"].lower()


def test_high_risk_query_triggers_guardrail(client):
    data = _ask(
        client, "I'm having severe chest pain and shortness of breath, what should I do?"
    )
    assert data["guardrail_triggered"] is True
    assert data["evidence_used"] == []
    assert "911" in data["answer"] or "emergency" in data["answer"].lower()


def test_vague_question_is_refused(client):
    data = _ask(client, "Is it bad?")
    assert data["evidence_sufficient"] is False
    assert data["guardrail_triggered"] is False


@pytest.mark.parametrize(
    "question",
    [
        "I have crushing chest pain",
        "I keep fainting",
        "I feel like I can't breathe",
        "my face is drooping and my speech is slurred",
    ],
)
def test_guardrail_patterns(question):
    assert check_high_risk(question).triggered is True


@pytest.mark.parametrize(
    "question",
    [
        "What lifestyle changes help with heart failure?",
        "How much salt should I eat?",
    ],
)
def test_guardrail_does_not_overtrigger(question):
    assert check_high_risk(question).triggered is False
