"""Smoke and unit tests for the Ingress API (Phase 1).

Tests run against the FastAPI TestClient — no Docker, no Ollama required.
Ollama calls are mocked via unittest.mock.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models import ClassifierResponse
from app.router import route

client = TestClient(app)


# ---------------------------------------------------------------------------
# Router unit tests (pure Python — no mocking needed)
# ---------------------------------------------------------------------------


def test_router_execution_maps_to_worker():
    assert route("execution") == "worker"


def test_router_decomposition_maps_to_worker():
    assert route("decomposition") == "worker"


def test_router_novel_reasoning_maps_to_worker():
    assert route("novel_reasoning") == "worker"


def test_router_ambiguous_maps_to_clarify():
    assert route("ambiguous") == "clarify"


def test_router_unknown_intent_maps_to_clarify():
    assert route("totally_unknown") == "clarify"


# ---------------------------------------------------------------------------
# Classifier schema validation (no network)
# ---------------------------------------------------------------------------


def test_classifier_response_valid():
    cr = ClassifierResponse(intent="execution", confidence=0.9)
    assert cr.intent == "execution"
    assert cr.confidence == 0.9


def test_classifier_response_rejects_invalid_intent():
    with pytest.raises(ValidationError):
        ClassifierResponse(intent="nonsense", confidence=0.5)


# ---------------------------------------------------------------------------
# Classifier behaviour (unit — patching _call_ollama)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_classifier_falls_back_on_network_error():
    """Ollama unreachable on both attempts → intent=ambiguous, confidence=0.0."""
    import httpx
    from app.classifier import classify

    with patch("app.classifier._call_ollama", side_effect=httpx.ConnectError("refused")):
        result = await classify("hello")

    assert result.intent == "ambiguous"
    assert result.confidence == 0.0


@pytest.mark.anyio
async def test_classifier_falls_back_on_invalid_json():
    """Ollama returns non-JSON on both attempts → intent=ambiguous, confidence=0.0."""
    from app.classifier import classify

    with patch("app.classifier._call_ollama", return_value="not json at all"):
        result = await classify("hello")

    assert result.intent == "ambiguous"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# /ingest happy path (mock Ollama)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_classify_execution():
    return AsyncMock(return_value=ClassifierResponse(intent="execution", confidence=0.95))


@pytest.fixture
def mock_worker_response():
    return AsyncMock(return_value="Here is the result.")


def test_ingest_happy_path(mock_classify_execution, mock_worker_response):
    with (
        patch("app.main.classify", mock_classify_execution),
        patch("app.main.generate", mock_worker_response),
    ):
        response = client.post("/ingest", json={"input": "Run a Python script"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "execution"
    assert body["confidence"] == 0.95
    assert body["response"] == "Here is the result."


def test_ingest_ambiguous_returns_clarification():
    mock_classify = AsyncMock(return_value=ClassifierResponse(intent="ambiguous", confidence=0.0))
    with patch("app.main.classify", mock_classify):
        response = client.post("/ingest", json={"input": "???"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "ambiguous"
    assert "clarify" in body["response"].lower() or "detail" in body["response"].lower()


def test_ingest_rejects_missing_input():
    response = client.post("/ingest", json={})
    assert response.status_code == 422


def test_ingest_curly_braces_in_input_do_not_crash(mock_classify_execution, mock_worker_response):
    """User input containing Python format placeholders must not raise KeyError."""
    with (
        patch("app.main.classify", mock_classify_execution),
        patch("app.main.generate", mock_worker_response),
    ):
        response = client.post("/ingest", json={"input": "what does {foo} mean in {bar}?"})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# /v1/chat/completions shim
# ---------------------------------------------------------------------------


def test_chat_completions_returns_200(mock_classify_execution, mock_worker_response):
    payload = {
        "model": "agentic",
        "messages": [{"role": "user", "content": "hello"}],
    }
    with (
        patch("app.main.classify", mock_classify_execution),
        patch("app.main.generate", mock_worker_response),
    ):
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "Here is the result."


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
