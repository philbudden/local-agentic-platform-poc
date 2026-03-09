"""Smoke and unit tests for CortX v0.3.0.

Tests run against the FastAPI TestClient — no Docker, no Ollama required.
Ollama calls are mocked via unittest.mock.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from coretex.interfaces.classifier import ClassificationResult
from distributions.cortx.main import app
from modules.router_simple.router import RouterSimple, ROUTES

_router = RouterSimple()
client = TestClient(app)

# ---------------------------------------------------------------------------
# Router unit tests (pure Python — no mocking needed)
# ---------------------------------------------------------------------------


def test_router_execution_maps_to_worker():
    assert _router.route("execution") == "worker"


def test_router_planning_maps_to_worker():
    assert _router.route("planning") == "worker"


def test_router_analysis_maps_to_worker():
    assert _router.route("analysis") == "worker"


def test_router_ambiguous_maps_to_clarify():
    assert _router.route("ambiguous") == "clarify"


def test_router_unknown_intent_maps_to_clarify():
    assert _router.route("totally_unknown") == "clarify"


# ---------------------------------------------------------------------------
# Classifier internal validation model (no network)
# ---------------------------------------------------------------------------


def test_classifier_response_valid():
    from modules.classifier_basic.classifier import _ClassifierResponse

    cr = _ClassifierResponse(intent="execution", confidence=0.9)
    assert cr.intent == "execution"
    assert cr.confidence == 0.9


def test_classifier_response_rejects_invalid_intent():
    from modules.classifier_basic.classifier import _ClassifierResponse

    with pytest.raises(ValidationError):
        _ClassifierResponse(intent="nonsense", confidence=0.5)


# ---------------------------------------------------------------------------
# Classifier behaviour (unit — patching _call_ollama)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_classifier_falls_back_on_network_error():
    """Ollama unreachable on both attempts → intent=ambiguous, confidence=0.0."""
    from modules.classifier_basic.classifier import ClassifierBasic

    classifier = ClassifierBasic()
    with patch(
        "modules.classifier_basic.classifier._call_ollama",
        side_effect=httpx.ConnectError("refused"),
    ):
        result = await classifier.classify("Compare quantum and classical computing")

    assert result.intent == "ambiguous"
    assert result.confidence == 0.0


@pytest.mark.anyio
async def test_classifier_falls_back_on_invalid_json():
    """Ollama returns non-JSON on both attempts → intent=ambiguous, confidence=0.0."""
    from modules.classifier_basic.classifier import ClassifierBasic

    classifier = ClassifierBasic()
    with patch(
        "modules.classifier_basic.classifier._call_ollama",
        return_value="not json at all",
    ):
        result = await classifier.classify("Compare quantum and classical computing")

    assert result.intent == "ambiguous"
    assert result.confidence == 0.0


@pytest.mark.anyio
async def test_classifier_parses_markdown_fenced_json():
    """_parse strips markdown code fences before parsing."""
    from modules.classifier_basic.classifier import _parse

    fenced = "```json\n{\"intent\": \"execution\", \"confidence\": 0.9}\n```"
    result = _parse(fenced)
    assert result is not None
    assert result.intent == "execution"


@pytest.mark.anyio
async def test_classifier_normalises_alias_intent():
    """_parse maps a known alias (e.g. 'creative_writing') to a valid intent."""
    from modules.classifier_basic.classifier import _parse

    result = _parse('{"intent": "creative_writing", "confidence": 0.8}')
    assert result is not None
    assert result.intent == "execution"


@pytest.mark.anyio
async def test_classifier_normalises_alternative_field_name():
    """_parse accepts 'category' as an alternative to 'intent'."""
    from modules.classifier_basic.classifier import _parse

    result = _parse('{"category": "execution", "confidence": 0.7}')
    assert result is not None
    assert result.intent == "execution"


@pytest.mark.anyio
async def test_classifier_normalises_capitalised_intent():
    """_parse lowercases the intent value before matching."""
    from modules.classifier_basic.classifier import _parse

    result = _parse('{"intent": "Execution", "confidence": 0.85}')
    assert result is not None
    assert result.intent == "execution"


# ---------------------------------------------------------------------------
# /ingest happy path (mock Ollama)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_classify_execution():
    return AsyncMock(return_value=ClassificationResult(intent="execution", confidence=0.95))


@pytest.fixture
def mock_worker_response():
    # Returns a proper JSON action envelope so the executor path is exercised.
    return AsyncMock(return_value='{"action": "respond", "content": "Here is the result."}')


def test_ingest_happy_path(mock_classify_execution, mock_worker_response):
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
        patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
    ):
        response = client.post("/ingest", json={"input": "Run a Python script"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "execution"
    assert body["confidence"] == 0.95
    assert body["response"] == "Here is the result."


def test_ingest_ambiguous_returns_clarification():
    mock_classify = AsyncMock(return_value=ClassificationResult(intent="ambiguous", confidence=0.0))
    with patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify):
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
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
        patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
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
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
        patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
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


def test_worker_uses_execution_prompt():
    """execution prompt enforces conciseness."""
    from modules.worker_llm.worker import _PROMPTS

    prompt = _PROMPTS["execution"].lower()
    assert "concise" in prompt or "150 words" in prompt


@pytest.mark.anyio
async def test_worker_uses_planning_prompt():
    """planning prompt requests numbered steps."""
    from modules.worker_llm.worker import _PROMPTS

    prompt = _PROMPTS["planning"].lower()
    assert "numbered" in prompt or "step" in prompt


@pytest.mark.anyio
async def test_worker_uses_analysis_prompt():
    """analysis prompt requests focused analytical response."""
    from modules.worker_llm.worker import _PROMPTS

    prompt = _PROMPTS["analysis"].lower()
    assert "analytical" in prompt or "insight" in prompt or "focused" in prompt


@pytest.mark.anyio
async def test_worker_unknown_intent_uses_fallback():
    """generate() falls back gracefully for unrecognised intent."""
    from modules.worker_llm.worker import _FALLBACK_PROMPT, _PROMPTS

    assert _FALLBACK_PROMPT == _PROMPTS["execution"]


# ---------------------------------------------------------------------------
# Phase 2: graceful worker failure handling
# ---------------------------------------------------------------------------


def test_ingest_worker_failure_returns_graceful_response():
    """If Ollama is unavailable during the worker call, return 200 with failure envelope."""
    mock_classify = AsyncMock(return_value=ClassificationResult(intent="execution", confidence=0.9))
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify),
        patch(
            "modules.worker_llm.worker.WorkerLLM.generate",
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
    mock_classify = AsyncMock(return_value=ClassificationResult(intent="execution", confidence=0.9))
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify),
        patch(
            "modules.worker_llm.worker.WorkerLLM.generate",
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
    """Response schema is intact after v0.3.0 refactor."""
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
        patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
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

    router = RouterSimple()
    with caplog.at_level(logging.WARNING, logger="modules.router_simple.router"):
        handler = router.route("totally_unknown", request_id="test-123")

    assert handler == "clarify"
    assert any("router_fallback" in r.message for r in caplog.records)
    assert any("totally_unknown" in r.message for r in caplog.records)


def test_router_known_intent_logs_intent_router(caplog):
    """route() emits an intent_router info log for every routing decision."""
    import logging

    router = RouterSimple()
    with caplog.at_level(logging.INFO, logger="modules.router_simple.router"):
        handler = router.route("execution", request_id="test-456", confidence=0.95)

    assert handler == "worker"
    assert any("intent_router" in r.message for r in caplog.records)


@pytest.mark.anyio
async def test_classifier_result_logged_for_prefix_match(caplog):
    """classify() emits classifier_result log even when prefix check short-circuits."""
    import logging

    from modules.classifier_basic.classifier import ClassifierBasic

    classifier = ClassifierBasic()
    with caplog.at_level(logging.INFO, logger="modules.classifier_basic.classifier"):
        await classifier.classify("Write a haiku", request_id="test-789")

    assert any("classifier_result" in r.message for r in caplog.records)
    assert any("prefix_match" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# v0.2.0: Tool Registry unit tests
# ---------------------------------------------------------------------------


def test_tool_registry_register_and_get():
    """Registered tool is retrievable by name."""
    from coretex.registry.tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="Return the input unchanged",
        input_schema={"text": "string"},
        function=lambda text: text,
    )
    tool = registry.get("echo")
    assert tool.name == "echo"


def test_tool_registry_duplicate_raises():
    """Registering the same name twice raises ValueError."""
    from coretex.registry.tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.register("dup", "desc", {}, lambda: None)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("dup", "desc", {}, lambda: None)


def test_tool_registry_unknown_tool_raises():
    """Getting an unregistered tool raises ValueError."""
    from coretex.registry.tool_registry import ToolRegistry

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Unknown component"):
        registry.get("nonexistent")


def test_tool_registry_list():
    """list() returns the names of all registered tools."""
    from coretex.registry.tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.register("a", "d", {}, lambda: None)
    registry.register("b", "d", {}, lambda: None)

    assert set(registry.list()) == {"a", "b"}


def test_tool_execute_calls_function():
    """Tool.execute() calls the underlying function with the provided args."""
    from coretex.registry.tool_registry import Tool

    def add(x: int, y: int) -> int:
        return x + y

    tool = Tool(name="add", description="add two numbers", input_schema={}, function=add)
    result = tool.execute({"x": 3, "y": 4})
    assert result == 7


# ---------------------------------------------------------------------------
# v0.2.0: AgentAction unit tests
# ---------------------------------------------------------------------------


def test_agent_action_from_dict_respond():
    """from_dict correctly parses a respond action."""
    from coretex.runtime.executor import AgentAction

    action = AgentAction.from_dict({"action": "respond", "content": "Hello"})
    assert action.action == "respond"
    assert action.content == "Hello"
    assert action.tool is None


def test_agent_action_from_dict_tool():
    """from_dict correctly parses a tool action."""
    from coretex.runtime.executor import AgentAction

    action = AgentAction.from_dict(
        {"action": "tool", "tool": "read_file", "args": {"path": "/tmp/x"}}
    )
    assert action.action == "tool"
    assert action.tool == "read_file"
    assert action.args == {"path": "/tmp/x"}


def test_agent_action_args_defaults_to_empty_dict():
    """args defaults to {} when absent from the dict."""
    from coretex.runtime.executor import AgentAction

    action = AgentAction.from_dict({"action": "respond", "content": "hi"})
    assert action.args == {}


# ---------------------------------------------------------------------------
# v0.2.0: ToolExecutor unit tests
# ---------------------------------------------------------------------------


def test_executor_respond_action_returns_content():
    """executor.execute() with action='respond' returns content directly."""
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.executor import AgentAction, ToolExecutor

    executor = ToolExecutor(ToolRegistry())
    action = AgentAction(action="respond", content="Direct reply")
    assert executor.execute(action) == "Direct reply"


def test_executor_tool_action_executes_tool():
    """executor.execute() with action='tool' calls the registered tool."""
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.executor import AgentAction, ToolExecutor

    registry = ToolRegistry()
    registry.register("upper", "uppercase text", {"text": "string"}, lambda text: text.upper())
    executor = ToolExecutor(registry)
    action = AgentAction(action="tool", tool="upper", args={"text": "hello"})
    assert executor.execute(action) == "HELLO"


def test_executor_unknown_action_raises():
    """executor.execute() raises ValueError for an unrecognised action type."""
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.executor import AgentAction, ToolExecutor

    executor = ToolExecutor(ToolRegistry())
    action = AgentAction(action="fly")

    with pytest.raises(ValueError, match="Unknown action type"):
        executor.execute(action)


def test_executor_unknown_tool_raises():
    """executor.execute() raises ValueError when the requested tool is not registered."""
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.executor import AgentAction, ToolExecutor

    executor = ToolExecutor(ToolRegistry())
    action = AgentAction(action="tool", tool="ghost", args={})

    with pytest.raises(ValueError, match="Unknown component"):
        executor.execute(action)


# ---------------------------------------------------------------------------
# v0.2.0: parse_agent_output unit tests
# ---------------------------------------------------------------------------


def test_parse_agent_output_valid_respond():
    """parse_agent_output correctly parses a respond envelope."""
    from coretex.runtime.executor import parse_agent_output

    action = parse_agent_output('{"action": "respond", "content": "hi"}')
    assert action.action == "respond"
    assert action.content == "hi"


def test_parse_agent_output_valid_tool():
    """parse_agent_output correctly parses a tool envelope."""
    from coretex.runtime.executor import parse_agent_output

    action = parse_agent_output(
        '{"action": "tool", "tool": "read_file", "args": {"path": "/tmp/f"}}'
    )
    assert action.action == "tool"
    assert action.tool == "read_file"


def test_parse_agent_output_invalid_json_raises():
    """parse_agent_output raises on non-JSON input."""
    import json

    from coretex.runtime.executor import parse_agent_output

    with pytest.raises(json.JSONDecodeError):
        parse_agent_output("this is plain text")


# ---------------------------------------------------------------------------
# v0.2.0: Filesystem tool unit tests
# ---------------------------------------------------------------------------


def test_filesystem_read_file_returns_content(tmp_path):
    """read_file returns the text content of an existing file."""
    from modules.tools_filesystem.filesystem import read_file

    f = tmp_path / "sample.txt"
    f.write_text("hello world")
    assert read_file(str(f)) == "hello world"


def test_filesystem_read_file_missing_returns_error_string():
    """read_file returns an error string when the file does not exist."""
    from modules.tools_filesystem.filesystem import read_file

    result = read_file("/nonexistent/path/file.txt")
    assert "File not found" in result


# ---------------------------------------------------------------------------
# v0.2.0: bootstrap_tools registration
# ---------------------------------------------------------------------------


def test_bootstrap_tools_registers_read_file():
    """bootstrap registers the read_file tool at module load."""
    from distributions.cortx.bootstrap import tool_registry

    assert "read_file" in tool_registry.list()


# ---------------------------------------------------------------------------
# v0.2.0: Worker prompt JSON instructions
# ---------------------------------------------------------------------------


def test_worker_execution_prompt_includes_json_instruction():
    """execution prompt now includes JSON format instruction."""
    from modules.worker_llm.worker import _PROMPTS

    prompt = _PROMPTS["execution"]
    assert '"action"' in prompt
    assert '"respond"' in prompt


def test_worker_planning_prompt_includes_json_instruction():
    """planning prompt includes JSON format instruction."""
    from modules.worker_llm.worker import _PROMPTS

    assert '"action"' in _PROMPTS["planning"]


def test_worker_analysis_prompt_includes_json_instruction():
    """analysis prompt includes JSON format instruction."""
    from modules.worker_llm.worker import _PROMPTS

    assert '"action"' in _PROMPTS["analysis"]


# ---------------------------------------------------------------------------
# v0.2.0: /ingest with JSON agent output (integration)
# ---------------------------------------------------------------------------


def test_ingest_with_json_respond_action(mock_classify_execution):
    """generate() returning a JSON respond envelope is correctly unwrapped."""
    json_output = '{"action": "respond", "content": "The answer is 42."}'
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
        patch("modules.worker_llm.worker.WorkerLLM.generate", AsyncMock(return_value=json_output)),
    ):
        response = client.post("/ingest", json={"input": "What is the answer?"})

    assert response.status_code == 200
    assert response.json()["response"] == "The answer is 42."


def test_ingest_with_tool_call_read_file(mock_classify_execution, tmp_path):
    """generate() returning a tool call JSON causes the file to be read."""
    test_file = tmp_path / "notes.txt"
    test_file.write_text("important notes")
    json_output = (
        '{"action": "tool", "tool": "read_file", "args": {"path": "' + str(test_file) + '"}}'
    )
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
        patch("modules.worker_llm.worker.WorkerLLM.generate", AsyncMock(return_value=json_output)),
    ):
        response = client.post("/ingest", json={"input": "Read my notes"})

    assert response.status_code == 200
    assert response.json()["response"] == "important notes"


def test_ingest_plain_text_fallback(mock_classify_execution):
    """generate() returning plain text (non-JSON) is returned as-is."""
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
        patch("modules.worker_llm.worker.WorkerLLM.generate", AsyncMock(return_value="Just some plain text")),
    ):
        response = client.post("/ingest", json={"input": "Tell me something"})

    assert response.status_code == 200
    assert response.json()["response"] == "Just some plain text"


def test_ingest_unknown_tool_returns_failure_response(mock_classify_execution):
    """Requesting an unregistered tool results in the worker failure response."""
    json_output = '{"action": "tool", "tool": "nonexistent_tool", "args": {}}'
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
        patch("modules.worker_llm.worker.WorkerLLM.generate", AsyncMock(return_value=json_output)),
    ):
        response = client.post("/ingest", json={"input": "Do something"})

    assert response.status_code == 200
    body = response.json()
    assert "unable to process" in body["response"].lower() or "sorry" in body["response"].lower()


def test_ingest_agent_selected_event_logged(caplog, mock_classify_execution, mock_worker_response):
    """event=worker_start is emitted when the worker is invoked."""
    import logging

    with caplog.at_level(logging.INFO, logger="coretex.runtime.pipeline"):
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
        ):
            client.post("/ingest", json={"input": "Run a task"})

    assert any("worker_start" in r.message for r in caplog.records)


def test_ingest_unexpected_tool_exception_returns_200_failure(mock_classify_execution):
    """A tool function raising an unexpected exception must NOT produce a 500."""
    from coretex.registry.tool_registry import Tool
    from distributions.cortx.bootstrap import tool_registry

    def _exploding_tool(**kwargs):
        raise RuntimeError("disk on fire")

    tool_registry._tools["exploding_tool"] = Tool(
        name="exploding_tool",
        description="always raises",
        input_schema={},
        function=_exploding_tool,
    )
    try:
        json_output = '{"action": "tool", "tool": "exploding_tool", "args": {}}'
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", AsyncMock(return_value=json_output)),
        ):
            response = client.post("/ingest", json={"input": "blow up"})

        assert response.status_code == 200
        body = response.json()
        assert "unable to process" in body["response"].lower() or "sorry" in body["response"].lower()
    finally:
        tool_registry._tools.pop("exploding_tool", None)


def test_ingest_tool_action_missing_tool_name_returns_failure(mock_classify_execution):
    """action='tool' with no 'tool' key must return a graceful failure response."""
    json_output = '{"action": "tool", "args": {}}'
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
        patch("modules.worker_llm.worker.WorkerLLM.generate", AsyncMock(return_value=json_output)),
    ):
        response = client.post("/ingest", json={"input": "call unnamed tool"})

    assert response.status_code == 200
    body = response.json()
    assert "unable to process" in body["response"].lower() or "sorry" in body["response"].lower()


def test_ingest_tool_execution_logs_carry_request_id(caplog, mock_classify_execution, tmp_path):
    """Tool execution log events must include the originating request_id."""
    import logging

    test_file = tmp_path / "data.txt"
    test_file.write_text("correlation check")
    json_output = (
        '{"action": "tool", "tool": "read_file", "args": {"path": "' + str(test_file) + '"}}'
    )

    with caplog.at_level(logging.INFO):
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", AsyncMock(return_value=json_output)),
        ):
            response = client.post("/ingest", json={"input": "Read data"})

    assert response.status_code == 200

    # At least one tool_execute event must carry a non-empty request_id.
    tool_execute_records = [r for r in caplog.records if "tool_execute" in r.message]
    assert tool_execute_records, "No tool_execute log event found"
    assert any("request_id=" in r.message for r in tool_execute_records), (
        "tool_execute events do not carry request_id"
    )


# ---------------------------------------------------------------------------
# Section 4.1 — Registry duplicate and unknown-lookup tests
# ---------------------------------------------------------------------------


def test_module_registry_duplicate_classifier_raises():
    """Registering the same classifier name twice raises ValueError."""
    from coretex.registry.module_registry import ModuleRegistry
    from coretex.interfaces.classifier import Classifier, ClassificationResult

    class _Dummy(Classifier):
        async def classify(self, text: str, request_id: str = "") -> ClassificationResult:
            return ClassificationResult(intent="execution", confidence=1.0)

    registry = ModuleRegistry()
    registry.register_classifier("dup", _Dummy())
    with pytest.raises(ValueError, match="already registered"):
        registry.register_classifier("dup", _Dummy())


def test_module_registry_duplicate_router_raises():
    """Registering the same router name twice raises ValueError."""
    from coretex.registry.module_registry import ModuleRegistry
    from coretex.interfaces.router import Router

    class _Dummy(Router):
        def route(self, intent: str, **kwargs: object) -> str:
            return "worker"

    registry = ModuleRegistry()
    registry.register_router("dup", _Dummy())
    with pytest.raises(ValueError, match="already registered"):
        registry.register_router("dup", _Dummy())


def test_module_registry_duplicate_worker_raises():
    """Registering the same worker name twice raises ValueError."""
    from coretex.registry.module_registry import ModuleRegistry
    from coretex.interfaces.worker import Worker

    class _Dummy(Worker):
        async def generate(self, text: str, intent: str = "", request_id: str = "") -> str:
            return "result"

    registry = ModuleRegistry()
    registry.register_worker("dup", _Dummy())
    with pytest.raises(ValueError, match="already registered"):
        registry.register_worker("dup", _Dummy())


def test_module_registry_unknown_classifier_raises():
    """Getting an unregistered classifier raises ValueError."""
    from coretex.registry.module_registry import ModuleRegistry

    registry = ModuleRegistry()
    with pytest.raises(ValueError, match="Unknown component"):
        registry.get_classifier("nonexistent")


def test_module_registry_unknown_router_raises():
    """Getting an unregistered router raises ValueError."""
    from coretex.registry.module_registry import ModuleRegistry

    registry = ModuleRegistry()
    with pytest.raises(ValueError, match="Unknown component"):
        registry.get_router("nonexistent")


def test_module_registry_unknown_worker_raises():
    """Getting an unregistered worker raises ValueError."""
    from coretex.registry.module_registry import ModuleRegistry

    registry = ModuleRegistry()
    with pytest.raises(ValueError, match="Unknown component"):
        registry.get_worker("nonexistent")


def test_module_registry_unknown_classifier_logs_lookup_failed(caplog):
    """get_classifier() emits event=registry_lookup_failed for unknown name."""
    import logging
    from coretex.registry.module_registry import ModuleRegistry

    registry = ModuleRegistry()
    with caplog.at_level(logging.ERROR, logger="coretex.registry.module_registry"):
        with pytest.raises(ValueError):
            registry.get_classifier("ghost")
    assert any("registry_lookup_failed" in r.message for r in caplog.records)


def test_model_registry_duplicate_raises():
    """Registering the same model provider twice raises ValueError."""
    from coretex.registry.model_registry import ModelProviderRegistry
    from coretex.interfaces.model_provider import ModelProvider

    class _Dummy(ModelProvider):
        async def generate(self, prompt: str, **kwargs: object) -> str:
            return ""
        async def chat(self, messages: list, **kwargs: object) -> str:
            return ""

    registry = ModelProviderRegistry()
    registry.register("dup", _Dummy())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("dup", _Dummy())


def test_model_registry_unknown_raises():
    """Getting an unregistered model provider raises ValueError."""
    from coretex.registry.model_registry import ModelProviderRegistry

    registry = ModelProviderRegistry()
    with pytest.raises(ValueError, match="Unknown component"):
        registry.get("nonexistent")


def test_model_registry_unknown_logs_lookup_failed(caplog):
    """get() emits event=registry_lookup_failed for unknown model provider."""
    import logging
    from coretex.registry.model_registry import ModelProviderRegistry

    registry = ModelProviderRegistry()
    with caplog.at_level(logging.ERROR, logger="coretex.registry.model_registry"):
        with pytest.raises(ValueError):
            registry.get("ghost")
    assert any("registry_lookup_failed" in r.message for r in caplog.records)


def test_pipeline_registry_duplicate_raises():
    """Registering the same pipeline name twice raises ValueError."""
    from coretex.registry.pipeline_registry import PipelineRegistry

    registry = PipelineRegistry()
    registry.register("dup", object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("dup", object())


def test_pipeline_registry_unknown_raises():
    """Getting an unregistered pipeline raises ValueError with pipeline-specific message."""
    from coretex.registry.pipeline_registry import PipelineRegistry

    registry = PipelineRegistry()
    with pytest.raises(ValueError, match="Unknown pipeline"):
        registry.get("nonexistent")


def test_tool_registry_lookup_failed_log(caplog):
    """get() emits event=registry_lookup_failed for unknown tool."""
    import logging
    from coretex.registry.tool_registry import ToolRegistry

    registry = ToolRegistry()
    with caplog.at_level(logging.ERROR, logger="coretex.registry.tool_registry"):
        with pytest.raises(ValueError):
            registry.get("ghost")
    assert any("registry_lookup_failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Section 4.2 — ModuleLoader validation tests
# ---------------------------------------------------------------------------


def test_module_loader_loads_valid_module(tmp_path, monkeypatch):
    """ModuleLoader.load() successfully registers a module with a valid register()."""
    import sys
    from coretex.registry.module_registry import ModuleRegistry
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.loader import ModuleLoader

    mod_dir = tmp_path / "mymod"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    (mod_dir / "module.py").write_text(
        "def register(module_registry, tool_registry, model_registry):\n"
        "    tool_registry.register('test_tool', 'desc', {}, lambda: 'ok')\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    mr = ModuleRegistry()
    tr = ToolRegistry()
    loader = ModuleLoader(mr, tr)
    loader.load("mymod.module")

    assert "test_tool" in tr.list()
    assert "mymod.module" in mr.list_loaded()


def test_module_loader_missing_register_raises(tmp_path, monkeypatch):
    """ModuleLoader.load() raises ValueError when module has no register() function."""
    import sys
    from coretex.registry.module_registry import ModuleRegistry
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.loader import ModuleLoader

    mod_dir = tmp_path / "noregmod"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    (mod_dir / "module.py").write_text("# no register function\n")

    monkeypatch.syspath_prepend(str(tmp_path))
    mr = ModuleRegistry()
    tr = ToolRegistry()
    loader = ModuleLoader(mr, tr)

    with pytest.raises(ValueError, match="no register\\(\\)"):
        loader.load("noregmod.module")


def test_module_loader_wrong_signature_raises(tmp_path, monkeypatch):
    """ModuleLoader.load() raises ValueError when register() has wrong signature."""
    import sys
    from coretex.registry.module_registry import ModuleRegistry
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.loader import ModuleLoader

    mod_dir = tmp_path / "badsigmod"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    (mod_dir / "module.py").write_text("def register(foo, bar):\n    pass\n")

    monkeypatch.syspath_prepend(str(tmp_path))
    mr = ModuleRegistry()
    tr = ToolRegistry()
    loader = ModuleLoader(mr, tr)

    with pytest.raises(ValueError, match="Invalid module register\\(\\) signature"):
        loader.load("badsigmod.module")


def test_module_loader_empty_registration_logs_warning(tmp_path, monkeypatch, caplog):
    """ModuleLoader.load() emits a warning when module registers no components."""
    import logging
    import sys
    from coretex.registry.module_registry import ModuleRegistry
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.loader import ModuleLoader

    mod_dir = tmp_path / "emptymod"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    (mod_dir / "module.py").write_text(
        "def register(module_registry, tool_registry, model_registry):\n    pass\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    mr = ModuleRegistry()
    tr = ToolRegistry()
    loader = ModuleLoader(mr, tr)

    with caplog.at_level(logging.WARNING, logger="coretex.runtime.loader"):
        loader.load("emptymod.module")

    assert any("module_registered_nothing" in r.message for r in caplog.records)


def test_module_loader_import_error_raises(tmp_path, monkeypatch):
    """ModuleLoader.load() raises ImportError for a non-existent module path."""
    from coretex.registry.module_registry import ModuleRegistry
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.loader import ModuleLoader

    mr = ModuleRegistry()
    tr = ToolRegistry()
    loader = ModuleLoader(mr, tr)

    with pytest.raises(ImportError):
        loader.load("definitely.does.not.exist")


def test_module_loader_load_all_emits_lifecycle_logs(tmp_path, monkeypatch, caplog):
    """load_all() emits module_loading_start and module_loading_complete events."""
    import logging
    from coretex.registry.module_registry import ModuleRegistry
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.loader import ModuleLoader

    mod_dir = tmp_path / "liftmod"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    (mod_dir / "module.py").write_text(
        "def register(module_registry, tool_registry, model_registry):\n    pass\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    mr = ModuleRegistry()
    tr = ToolRegistry()
    loader = ModuleLoader(mr, tr)

    with caplog.at_level(logging.INFO, logger="coretex.runtime.loader"):
        loader.load_all(["liftmod.module"])

    messages = [r.message for r in caplog.records]
    assert any("module_loading_start" in m for m in messages)
    assert any("module_loading_complete" in m for m in messages)


# ---------------------------------------------------------------------------
# Section 4.3 — ToolExecutor additional tests
# ---------------------------------------------------------------------------


def test_executor_tool_action_missing_tool_name_raises():
    """execute() raises ValueError when action='tool' but tool name is absent."""
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.executor import AgentAction, ToolExecutor

    executor = ToolExecutor(ToolRegistry())
    action = AgentAction(action="tool", tool=None, args={})

    with pytest.raises(ValueError):
        executor.execute(action)


def test_executor_respond_action_none_content_returns_none():
    """execute() with action='respond' and content=None returns None."""
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.executor import AgentAction, ToolExecutor

    executor = ToolExecutor(ToolRegistry())
    action = AgentAction(action="respond", content=None)
    assert executor.execute(action) is None


def test_executor_tool_runtime_exception_propagates():
    """A tool function raising an exception causes execute() to propagate it."""
    from coretex.registry.tool_registry import ToolRegistry
    from coretex.runtime.executor import AgentAction, ToolExecutor

    registry = ToolRegistry()
    registry.register("boom", "explodes", {}, lambda: (_ for _ in ()).throw(RuntimeError("kaboom")))
    executor = ToolExecutor(registry)
    action = AgentAction(action="tool", tool="boom", args={})

    with pytest.raises(RuntimeError, match="kaboom"):
        executor.execute(action)


# ---------------------------------------------------------------------------
# Section 4.4 — Pipeline failure tests (mock scenarios)
# ---------------------------------------------------------------------------


def test_pipeline_classifier_http_failure_returns_clarification():
    """Classifier HTTP failure falls back to intent=ambiguous and clarification response."""
    import httpx
    with patch(
        "modules.classifier_basic.classifier.ClassifierBasic.classify",
        side_effect=httpx.ConnectError("refused"),
    ):
        response = client.post("/ingest", json={"input": "Do something"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "ambiguous"
    assert body["confidence"] == 0.0


def test_pipeline_worker_http_failure_returns_graceful_response():
    """Worker HTTP failure returns 200 with failure response."""
    import httpx
    mock_classify = AsyncMock(
        return_value=__import__(
            "coretex.interfaces.classifier",
            fromlist=["ClassificationResult"],
        ).ClassificationResult(intent="execution", confidence=0.9)
    )
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify),
        patch(
            "modules.worker_llm.worker.WorkerLLM.generate",
            side_effect=httpx.HTTPStatusError("error", request=None, response=None),
        ),
    ):
        response = client.post("/ingest", json={"input": "Do something"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "ambiguous"
    assert len(body["response"]) > 0


def test_pipeline_invalid_json_output_treated_as_plain_text(mock_classify_execution):
    """Worker returning plain text (non-JSON) is returned as-is without 500."""
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
        patch(
            "modules.worker_llm.worker.WorkerLLM.generate",
            AsyncMock(return_value="This is plain text output"),
        ),
    ):
        response = client.post("/ingest", json={"input": "Tell me something"})

    assert response.status_code == 200
    assert response.json()["response"] == "This is plain text output"


def test_pipeline_tool_lookup_failure_returns_worker_failure(mock_classify_execution):
    """Requesting an unregistered tool results in the worker failure response."""
    json_output = '{"action": "tool", "tool": "nonexistent_tool", "args": {}}'
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
        patch("modules.worker_llm.worker.WorkerLLM.generate", AsyncMock(return_value=json_output)),
    ):
        response = client.post("/ingest", json={"input": "Do something"})

    assert response.status_code == 200
    body = response.json()
    assert "unable to process" in body["response"].lower() or "sorry" in body["response"].lower()


def test_pipeline_tool_runtime_exception_returns_worker_failure(mock_classify_execution):
    """Tool raising a runtime exception returns worker failure response, not 500."""
    from coretex.registry.tool_registry import Tool
    from distributions.cortx.bootstrap import tool_registry

    def _boom(**kwargs: object) -> None:
        raise RuntimeError("unexpected error")

    tool_registry._tools["runtime_failure_tool"] = Tool(
        name="runtime_failure_tool",
        description="always fails",
        input_schema={},
        function=_boom,
    )
    try:
        json_output = '{"action": "tool", "tool": "runtime_failure_tool", "args": {}}'
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", AsyncMock(return_value=json_output)),
        ):
            response = client.post("/ingest", json={"input": "Trigger failure"})

        assert response.status_code == 200
        body = response.json()
        assert "unable to process" in body["response"].lower() or "sorry" in body["response"].lower()
    finally:
        tool_registry._tools.pop("runtime_failure_tool", None)


# ---------------------------------------------------------------------------
# Section 4.5 — Logging tests
# ---------------------------------------------------------------------------


def test_pipeline_logs_request_received(caplog, mock_classify_execution, mock_worker_response):
    """event=request_received is emitted at the start of every pipeline run."""
    import logging

    with caplog.at_level(logging.INFO, logger="coretex.runtime.pipeline"):
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
        ):
            client.post("/ingest", json={"input": "Hello"})

    assert any("request_received" in r.message for r in caplog.records)


def test_pipeline_logs_classifier_complete(caplog, mock_classify_execution, mock_worker_response):
    """event=classifier_complete is emitted after classification."""
    import logging

    with caplog.at_level(logging.INFO, logger="coretex.runtime.pipeline"):
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
        ):
            client.post("/ingest", json={"input": "Hello"})

    assert any("classifier_complete" in r.message for r in caplog.records)


def test_pipeline_logs_router_selected(caplog, mock_classify_execution, mock_worker_response):
    """event=router_selected is emitted after routing."""
    import logging

    with caplog.at_level(logging.INFO, logger="coretex.runtime.pipeline"):
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
        ):
            client.post("/ingest", json={"input": "Hello"})

    assert any("router_selected" in r.message for r in caplog.records)


def test_pipeline_logs_worker_complete(caplog, mock_classify_execution, mock_worker_response):
    """event=worker_complete is emitted after the worker finishes."""
    import logging

    with caplog.at_level(logging.INFO, logger="coretex.runtime.pipeline"):
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
        ):
            client.post("/ingest", json={"input": "Hello"})

    assert any("worker_complete" in r.message for r in caplog.records)


def test_pipeline_logs_request_complete(caplog, mock_classify_execution, mock_worker_response):
    """event=request_complete is emitted at the end of every pipeline run."""
    import logging

    with caplog.at_level(logging.INFO, logger="coretex.runtime.pipeline"):
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
        ):
            client.post("/ingest", json={"input": "Hello"})

    assert any("request_complete" in r.message for r in caplog.records)


def test_pipeline_request_complete_contains_duration_ms(caplog, mock_classify_execution, mock_worker_response):
    """event=request_complete includes total_latency_ms."""
    import logging

    with caplog.at_level(logging.INFO, logger="coretex.runtime.pipeline"):
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
        ):
            client.post("/ingest", json={"input": "Hello"})

    complete_records = [r for r in caplog.records if "request_complete" in r.message]
    assert complete_records
    assert any("total_latency_ms" in r.message for r in complete_records)


def test_pipeline_classifier_complete_contains_duration_ms(caplog, mock_classify_execution, mock_worker_response):
    """event=classifier_complete includes duration_ms."""
    import logging

    with caplog.at_level(logging.INFO, logger="coretex.runtime.pipeline"):
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
        ):
            client.post("/ingest", json={"input": "Hello"})

    classifier_records = [r for r in caplog.records if "classifier_complete" in r.message]
    assert classifier_records
    assert any("duration_ms" in r.message for r in classifier_records)


def test_pipeline_classifier_failure_logs_event(caplog):
    """event=pipeline_classifier_failure is emitted when classifier raises HTTP error."""
    import logging
    import httpx

    with caplog.at_level(logging.ERROR, logger="coretex.runtime.pipeline"):
        with patch(
            "modules.classifier_basic.classifier.ClassifierBasic.classify",
            side_effect=httpx.ConnectError("refused"),
        ):
            client.post("/ingest", json={"input": "Something"})

    assert any("pipeline_classifier_failure" in r.message for r in caplog.records)


def test_pipeline_worker_failure_logs_event(caplog):
    """event=pipeline_worker_failure is emitted when worker raises HTTP error."""
    import logging
    import httpx

    mock_classify = AsyncMock(
        return_value=__import__(
            "coretex.interfaces.classifier",
            fromlist=["ClassificationResult"],
        ).ClassificationResult(intent="execution", confidence=0.9)
    )
    with caplog.at_level(logging.ERROR, logger="coretex.runtime.pipeline"):
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify),
            patch(
                "modules.worker_llm.worker.WorkerLLM.generate",
                side_effect=httpx.ConnectError("refused"),
            ),
        ):
            client.post("/ingest", json={"input": "Something"})

    assert any("pipeline_worker_failure" in r.message for r in caplog.records)


def test_pipeline_tool_failure_logs_event(caplog, mock_classify_execution):
    """event=pipeline_tool_failure is emitted when tool execution raises."""
    import logging
    from coretex.registry.tool_registry import Tool
    from distributions.cortx.bootstrap import tool_registry

    def _boom(**kwargs: object) -> None:
        raise RuntimeError("tool error")

    tool_registry._tools["log_fail_tool"] = Tool(
        name="log_fail_tool", description="fails", input_schema={}, function=_boom
    )
    try:
        json_output = '{"action": "tool", "tool": "log_fail_tool", "args": {}}'
        with caplog.at_level(logging.ERROR, logger="coretex.runtime.pipeline"):
            with (
                patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
                patch("modules.worker_llm.worker.WorkerLLM.generate", AsyncMock(return_value=json_output)),
            ):
                client.post("/ingest", json={"input": "Fail"})

        assert any("pipeline_tool_failure" in r.message for r in caplog.records)
    finally:
        tool_registry._tools.pop("log_fail_tool", None)


# ---------------------------------------------------------------------------
# Section 4 — ExecutionContext tests
# ---------------------------------------------------------------------------


def test_execution_context_has_timestamp():
    """ExecutionContext.timestamp is a float (wall-clock time)."""
    import time
    from coretex.runtime.context import ExecutionContext

    before = time.time()
    ctx = ExecutionContext(user_input="hello")
    after = time.time()

    assert isinstance(ctx.timestamp, float)
    assert before <= ctx.timestamp <= after


def test_execution_context_metadata_defaults_to_none():
    """ExecutionContext.metadata defaults to None."""
    from coretex.runtime.context import ExecutionContext

    ctx = ExecutionContext(user_input="hello")
    assert ctx.metadata is None


def test_execution_context_metadata_can_be_set():
    """ExecutionContext.metadata can hold an arbitrary dict."""
    from coretex.runtime.context import ExecutionContext

    ctx = ExecutionContext(user_input="hello", metadata={"source": "test"})
    assert ctx.metadata == {"source": "test"}


# ---------------------------------------------------------------------------
# Section 4 — Router debug_router tests
# ---------------------------------------------------------------------------


def test_router_debug_decision_logged_when_debug_router_enabled(caplog):
    """event=router_decision is emitted at DEBUG level when debug_router=True."""
    import logging
    from unittest.mock import patch as _patch
    from modules.router_simple.router import RouterSimple

    router = RouterSimple()
    with _patch("modules.router_simple.router.settings") as mock_settings:
        mock_settings.debug_router = True
        with caplog.at_level(logging.DEBUG, logger="modules.router_simple.router"):
            router.route("execution", request_id="dbg-test")

    assert any("router_decision" in r.message for r in caplog.records)


def test_router_debug_decision_not_logged_when_debug_router_disabled(caplog):
    """event=router_decision is NOT emitted when debug_router=False."""
    import logging
    from unittest.mock import patch as _patch
    from modules.router_simple.router import RouterSimple

    router = RouterSimple()
    with _patch("modules.router_simple.router.settings") as mock_settings:
        mock_settings.debug_router = False
        with caplog.at_level(logging.DEBUG, logger="modules.router_simple.router"):
            router.route("execution", request_id="nodbg-test")

    assert not any("router_decision" in r.message for r in caplog.records)



# ---------------------------------------------------------------------------
# v0.4.0: PipelineStep unit tests
# ---------------------------------------------------------------------------


def test_pipeline_step_valid_classifier_type():
    """PipelineStep accepts 'classifier' as a valid component_type."""
    from coretex.runtime.pipeline import PipelineStep

    step = PipelineStep(component_type="classifier", name="classifier_basic")
    assert step.component_type == "classifier"
    assert step.name == "classifier_basic"


def test_pipeline_step_valid_router_type():
    """PipelineStep accepts 'router' as a valid component_type."""
    from coretex.runtime.pipeline import PipelineStep

    step = PipelineStep(component_type="router", name="router_simple")
    assert step.component_type == "router"


def test_pipeline_step_valid_worker_type():
    """PipelineStep accepts 'worker' as a valid component_type."""
    from coretex.runtime.pipeline import PipelineStep

    step = PipelineStep(component_type="worker", name="worker_llm")
    assert step.component_type == "worker"


def test_pipeline_step_valid_tool_executor_type():
    """PipelineStep accepts 'tool_executor' as a valid component_type."""
    from coretex.runtime.pipeline import PipelineStep

    step = PipelineStep(component_type="tool_executor", name="tool_executor")
    assert step.component_type == "tool_executor"


def test_pipeline_step_invalid_type_raises():
    """PipelineStep raises ValueError for an unrecognised component_type."""
    from coretex.runtime.pipeline import PipelineStep

    with pytest.raises(ValueError, match="Invalid step component_type"):
        PipelineStep(component_type="planner", name="planner_basic")


def test_pipeline_step_empty_type_raises():
    """PipelineStep raises ValueError for an empty component_type string."""
    from coretex.runtime.pipeline import PipelineStep

    with pytest.raises(ValueError, match="Invalid step component_type"):
        PipelineStep(component_type="", name="something")


# ---------------------------------------------------------------------------
# v0.4.0: PipelineDefinition unit tests
# ---------------------------------------------------------------------------


def test_pipeline_definition_name_and_steps():
    """PipelineDefinition stores name and steps correctly."""
    from coretex.runtime.pipeline import PipelineDefinition, PipelineStep

    steps = [
        PipelineStep(component_type="classifier", name="classifier_basic"),
        PipelineStep(component_type="router", name="router_simple"),
    ]
    defn = PipelineDefinition(name="test_pipeline", steps=steps)
    assert defn.name == "test_pipeline"
    assert len(defn.steps) == 2


def test_pipeline_definition_get_step_returns_matching_step():
    """get_step() returns the first step with the requested component_type."""
    from coretex.runtime.pipeline import PipelineDefinition, PipelineStep

    steps = [
        PipelineStep(component_type="classifier", name="my_classifier"),
        PipelineStep(component_type="worker", name="my_worker"),
    ]
    defn = PipelineDefinition(name="custom", steps=steps)
    step = defn.get_step("classifier")
    assert step is not None
    assert step.name == "my_classifier"


def test_pipeline_definition_get_step_returns_none_for_missing():
    """get_step() returns None when no step with the requested type exists."""
    from coretex.runtime.pipeline import PipelineDefinition, PipelineStep

    defn = PipelineDefinition(
        name="minimal",
        steps=[PipelineStep(component_type="classifier", name="c")],
    )
    assert defn.get_step("worker") is None


def test_pipeline_definition_empty_steps():
    """PipelineDefinition can be created with an empty steps list."""
    from coretex.runtime.pipeline import PipelineDefinition

    defn = PipelineDefinition(name="empty")
    assert defn.steps == []


def test_make_default_pipeline_has_correct_name():
    """make_default_pipeline() returns a pipeline named 'default'."""
    from coretex.runtime.pipeline import make_default_pipeline, DEFAULT_PIPELINE_NAME

    defn = make_default_pipeline()
    assert defn.name == DEFAULT_PIPELINE_NAME


def test_make_default_pipeline_has_four_steps():
    """make_default_pipeline() returns a pipeline with exactly four steps."""
    from coretex.runtime.pipeline import make_default_pipeline

    defn = make_default_pipeline()
    assert len(defn.steps) == 4


def test_make_default_pipeline_step_names():
    """make_default_pipeline() uses the standard module names."""
    from coretex.runtime.pipeline import make_default_pipeline

    defn = make_default_pipeline()
    types = {s.component_type: s.name for s in defn.steps}
    assert types["classifier"] == "classifier_basic"
    assert types["router"] == "router_simple"
    assert types["worker"] == "worker_llm"
    assert types["tool_executor"] == "tool_executor"


# ---------------------------------------------------------------------------
# v0.4.0: PipelineRegistry enhanced tests
# ---------------------------------------------------------------------------


def test_pipeline_registry_duplicate_raises_pipeline_specific_message():
    """Registering the same pipeline name twice raises ValueError with pipeline-specific message."""
    from coretex.runtime.pipeline import PipelineDefinition
    from coretex.registry.pipeline_registry import PipelineRegistry

    registry = PipelineRegistry()
    defn = PipelineDefinition(name="dup_test")
    registry.register("dup_test", defn)

    with pytest.raises(ValueError, match="Pipeline already registered: dup_test"):
        registry.register("dup_test", defn)


def test_pipeline_registry_unknown_raises_pipeline_specific_message():
    """Getting an unregistered pipeline raises ValueError with pipeline-specific message."""
    from coretex.registry.pipeline_registry import PipelineRegistry

    registry = PipelineRegistry()

    with pytest.raises(ValueError, match="Unknown pipeline: missing"):
        registry.get("missing")


def test_pipeline_registry_list_returns_registered_names():
    """list() returns all registered pipeline names."""
    from coretex.runtime.pipeline import PipelineDefinition
    from coretex.registry.pipeline_registry import PipelineRegistry

    registry = PipelineRegistry()
    registry.register("alpha", PipelineDefinition(name="alpha"))
    registry.register("beta", PipelineDefinition(name="beta"))

    names = registry.list()
    assert set(names) == {"alpha", "beta"}


def test_pipeline_registry_logs_lookup_failed(caplog):
    """get() emits event=registry_lookup_failed for unknown pipeline."""
    import logging
    from coretex.registry.pipeline_registry import PipelineRegistry

    registry = PipelineRegistry()
    with caplog.at_level(logging.ERROR, logger="coretex.registry.pipeline_registry"):
        with pytest.raises(ValueError):
            registry.get("ghost_pipeline")
    assert any("registry_lookup_failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# v0.4.0: PipelineRunner with PipelineDefinition
# ---------------------------------------------------------------------------


def test_pipeline_runner_uses_classifier_from_pipeline_definition(mock_classify_execution, mock_worker_response):
    """PipelineRunner respects the classifier name from the PipelineDefinition."""
    from coretex.runtime.pipeline import PipelineDefinition, PipelineStep, PipelineRunner
    from distributions.cortx.bootstrap import module_registry, tool_registry

    defn = PipelineDefinition(
        name="test",
        steps=[
            PipelineStep(component_type="classifier", name="classifier_basic"),
            PipelineStep(component_type="router", name="router_simple"),
            PipelineStep(component_type="worker", name="worker_llm"),
            PipelineStep(component_type="tool_executor", name="tool_executor"),
        ],
    )
    runner = PipelineRunner(
        module_registry=module_registry,
        tool_registry=tool_registry,
        pipeline=defn,
    )
    assert runner._classifier_name == "classifier_basic"
    assert runner._router_name == "router_simple"
    assert runner._worker_name == "worker_llm"


def test_pipeline_runner_default_pipeline_when_none_provided():
    """PipelineRunner uses the default pipeline when pipeline=None."""
    from coretex.runtime.pipeline import DEFAULT_PIPELINE_NAME, PipelineRunner
    from distributions.cortx.bootstrap import module_registry, tool_registry

    runner = PipelineRunner(
        module_registry=module_registry,
        tool_registry=tool_registry,
        pipeline=None,
    )
    assert runner._pipeline.name == DEFAULT_PIPELINE_NAME
    assert runner._classifier_name == "classifier_basic"


def test_default_pipeline_registered_in_bootstrap():
    """Bootstrap registers the default pipeline in the PipelineRegistry."""
    from distributions.cortx.bootstrap import pipeline_registry

    names = pipeline_registry.list()
    assert "default" in names


def test_pipeline_selected_event_logged(caplog, mock_classify_execution, mock_worker_response):
    """event=pipeline_selected is emitted at the start of every pipeline run."""
    import logging

    with caplog.at_level(logging.INFO, logger="coretex.runtime.pipeline"):
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
        ):
            client.post("/ingest", json={"input": "Hello"})

    assert any("pipeline_selected" in r.message for r in caplog.records)


def test_pipeline_selected_event_contains_pipeline_name(caplog, mock_classify_execution, mock_worker_response):
    """event=pipeline_selected log includes the pipeline name."""
    import logging

    with caplog.at_level(logging.INFO, logger="coretex.runtime.pipeline"):
        with (
            patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
            patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
        ):
            client.post("/ingest", json={"input": "Hello"})

    selected_records = [r for r in caplog.records if "pipeline_selected" in r.message]
    assert selected_records
    assert any("pipeline=default" in r.message for r in selected_records)


@pytest.mark.anyio
async def test_pipeline_runner_custom_pipeline_executes_full_request(mock_classify_execution, mock_worker_response):
    """A PipelineRunner built from a custom PipelineDefinition completes a full request."""
    from coretex.runtime.pipeline import PipelineDefinition, PipelineStep, PipelineRunner
    from coretex.runtime.context import ExecutionContext
    from distributions.cortx.bootstrap import module_registry, tool_registry

    custom = PipelineDefinition(
        name="custom_test",
        steps=[
            PipelineStep(component_type="classifier", name="classifier_basic"),
            PipelineStep(component_type="router", name="router_simple"),
            PipelineStep(component_type="worker", name="worker_llm"),
            PipelineStep(component_type="tool_executor", name="tool_executor"),
        ],
    )
    runner = PipelineRunner(
        module_registry=module_registry,
        tool_registry=tool_registry,
        pipeline=custom,
    )
    ctx = ExecutionContext(user_input="test input")
    with (
        patch("modules.classifier_basic.classifier.ClassifierBasic.classify", mock_classify_execution),
        patch("modules.worker_llm.worker.WorkerLLM.generate", mock_worker_response),
    ):
        response_text, intent, confidence = await runner.run(ctx)

    assert response_text == "Here is the result."
    assert intent == "execution"
    assert confidence == 0.95
