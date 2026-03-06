"""Smoke and unit tests for the Ingress API (Phase 1 + Phase 2).

Tests run against the FastAPI TestClient — no Docker, no Ollama required.
Ollama calls are mocked via unittest.mock.
"""

from unittest.mock import AsyncMock, patch

import httpx
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


def test_router_planning_maps_to_worker():
    assert route("planning") == "worker"


def test_router_analysis_maps_to_worker():
    assert route("analysis") == "worker"


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
        result = await classify("Compare quantum and classical computing")

    assert result.intent == "ambiguous"
    assert result.confidence == 0.0


@pytest.mark.anyio
async def test_classifier_falls_back_on_invalid_json():
    """Ollama returns non-JSON on both attempts → intent=ambiguous, confidence=0.0."""
    from app.classifier import classify

    with patch("app.classifier._call_ollama", return_value="not json at all"):
        result = await classify("Compare quantum and classical computing")

    assert result.intent == "ambiguous"
    assert result.confidence == 0.0


@pytest.mark.anyio
async def test_classifier_parses_markdown_fenced_json():
    """_parse strips markdown code fences before parsing."""
    from app.classifier import _parse

    fenced = "```json\n{\"intent\": \"execution\", \"confidence\": 0.9}\n```"
    result = _parse(fenced)
    assert result is not None
    assert result.intent == "execution"


@pytest.mark.anyio
async def test_classifier_normalises_alias_intent():
    """_parse maps a known alias (e.g. 'creative_writing') to a valid intent."""
    from app.classifier import _parse

    result = _parse('{"intent": "creative_writing", "confidence": 0.8}')
    assert result is not None
    assert result.intent == "execution"


@pytest.mark.anyio
async def test_classifier_normalises_alternative_field_name():
    """_parse accepts 'category' as an alternative to 'intent'."""
    from app.classifier import _parse

    result = _parse('{"category": "execution", "confidence": 0.7}')
    assert result is not None
    assert result.intent == "execution"


@pytest.mark.anyio
async def test_classifier_normalises_capitalised_intent():
    """_parse lowercases the intent value before matching."""
    from app.classifier import _parse

    result = _parse('{"intent": "Execution", "confidence": 0.85}')
    assert result is not None
    assert result.intent == "execution"


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


def test_ingest_rejects_empty_string_input():
    response = client.post("/ingest", json={"input": ""})
    assert response.status_code == 422


def test_ingest_rejects_whitespace_only_input():
    response = client.post("/ingest", json={"input": "   "})
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


def test_chat_completions_empty_messages_returns_clarification():
    """No user messages in the request must not crash — return clarification."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "agentic", "messages": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert len(body["choices"][0]["message"]["content"]) > 0


def test_chat_completions_whitespace_only_message_returns_clarification():
    """Whitespace-only user content must not crash — return clarification."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "agentic", "messages": [{"role": "user", "content": "   "}]},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models_returns_agentic():
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert any(m["id"] == "agentic" for m in body["data"])


# ---------------------------------------------------------------------------
# Phase 2: intent-aware worker prompts
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_worker_uses_execution_prompt():
    """execution prompt enforces conciseness."""
    from app.worker import _PROMPTS

    prompt = _PROMPTS["execution"].lower()
    assert "concise" in prompt or "150 words" in prompt


@pytest.mark.anyio
async def test_worker_uses_planning_prompt():
    """planning prompt requests numbered steps."""
    from app.worker import _PROMPTS

    prompt = _PROMPTS["planning"].lower()
    assert "numbered" in prompt or "step" in prompt


@pytest.mark.anyio
async def test_worker_uses_analysis_prompt():
    """analysis prompt requests focused analytical response."""
    from app.worker import _PROMPTS

    prompt = _PROMPTS["analysis"].lower()
    assert "analytical" in prompt or "insight" in prompt or "focused" in prompt


@pytest.mark.anyio
async def test_worker_unknown_intent_uses_fallback():
    """generate() falls back gracefully for unrecognised intent."""
    from app.worker import _FALLBACK_PROMPT, _PROMPTS

    assert _FALLBACK_PROMPT == _PROMPTS["execution"]


# ---------------------------------------------------------------------------
# Phase 2: graceful worker failure handling
# ---------------------------------------------------------------------------


def test_ingest_worker_failure_returns_graceful_response():
    """If Ollama is unavailable during the worker call, return 200 with failure envelope."""
    mock_classify = AsyncMock(return_value=ClassifierResponse(intent="execution", confidence=0.9))
    with (
        patch("app.main.classify", mock_classify),
        patch(
            "app.main.generate",
            side_effect=httpx.ConnectError("refused"),
        ),
    ):
        response = client.post("/ingest", json={"input": "Run a script"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "ambiguous"
    assert body["confidence"] == 0.0
    assert len(body["response"]) > 0


def test_ingest_worker_timeout_returns_graceful_response():
    """Worker timeout returns 200 with failure envelope rather than 500."""
    mock_classify = AsyncMock(return_value=ClassifierResponse(intent="execution", confidence=0.9))
    with (
        patch("app.main.classify", mock_classify),
        patch(
            "app.main.generate",
            side_effect=httpx.TimeoutException("timed out"),
        ),
    ):
        response = client.post("/ingest", json={"input": "Do something slow"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "ambiguous"
    assert body["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Phase 3: observability — /debug/routes, correlation IDs, router logging
# ---------------------------------------------------------------------------


def test_debug_routes_returns_routing_table():
    """GET /debug/routes returns the intent→handler mapping."""
    response = client.get("/debug/routes")
    assert response.status_code == 200
    body = response.json()
    assert "routes" in body
    routes = body["routes"]
    assert routes["execution"] == "worker"
    assert routes["planning"] == "worker"
    assert routes["analysis"] == "worker"
    assert routes["ambiguous"] == "clarify"


def test_ingest_response_contains_expected_fields(mock_classify_execution, mock_worker_response):
    """Response schema is intact after Phase 3 wiring changes."""
    with (
        patch("app.main.classify", mock_classify_execution),
        patch("app.main.generate", mock_worker_response),
    ):
        response = client.post("/ingest", json={"input": "Write a poem"})
    assert response.status_code == 200
    body = response.json()
    assert "intent" in body
    assert "confidence" in body
    assert "response" in body


def test_router_unknown_intent_logs_fallback(caplog):
    """route() emits a router_fallback warning for unrecognised intents."""
    import logging

    from app.router import route

    with caplog.at_level(logging.WARNING, logger="app.router"):
        handler = route("totally_unknown", request_id="test-123")

    assert handler == "clarify"
    assert any("router_fallback" in r.message for r in caplog.records)
    assert any("totally_unknown" in r.message for r in caplog.records)


def test_router_known_intent_logs_intent_router(caplog):
    """route() emits an intent_router info log for every routing decision."""
    import logging

    from app.router import route

    with caplog.at_level(logging.INFO, logger="app.router"):
        handler = route("execution", request_id="test-456", confidence=0.95)

    assert handler == "worker"
    assert any("intent_router" in r.message for r in caplog.records)


@pytest.mark.anyio
async def test_classifier_result_logged_for_prefix_match(caplog):
    """classify() emits classifier_result log even when prefix check short-circuits."""
    import logging

    from app.classifier import classify

    with caplog.at_level(logging.INFO, logger="app.classifier"):
        await classify("Write a haiku", request_id="test-789")

    assert any("classifier_result" in r.message for r in caplog.records)
    assert any("prefix_match" in r.message for r in caplog.records)
