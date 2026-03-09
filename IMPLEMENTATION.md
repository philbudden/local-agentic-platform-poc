# Implementation Description — COREtex Runtime Platform

> This document describes the current state of the implementation as of v0.3.x (Stabilisation).
> It is intended to be read as context by an LLM when planning future phases of the project.

---

## Purpose

COREtex is a **local agentic runtime platform** that routes user requests through a two-stage LLM pipeline (classifier → worker) using locally-hosted models via [Ollama](https://ollama.com), then passes the agent's JSON output through a deterministic tool execution layer.

v0.3.0 restructured the codebase from a monolithic `app/` package into a **modular runtime** with three distinct layers: the `coretex/` runtime package (interfaces, registries, execution engine), `modules/` (pluggable component implementations), and `distributions/` (assembled applications).

v0.3.x Stabilisation hardened the runtime with: explicit pipeline failure categories and graceful handling, full structured logging lifecycle, registry validation with consistent error messages, ModuleLoader signature validation and empty-registration detection, `load_all()` lifecycle events, router debug logging, and expanded test coverage to 106 tests.

---

## High-Level Architecture

```
User (browser)
  └─► OpenWebUI  (port 3000)
        └─► POST /v1/chat/completions  ← OpenAI-compatible shim
              └─► POST /ingest         ← distribution entry point (cortx)
                    └─► PipelineRunner.run(ExecutionContext)
                          ├─► ClassifierBasic.classify()  → ClassificationResult(intent, confidence)
                          ├─► RouterSimple.route(intent)  → handler name
                          └─► WorkerLLM.generate()        → JSON action envelope
                                      │
                               parse_agent_output          → AgentAction
                                      │
                               ToolExecutor.execute        → tool result or direct content
```

- **Ollama** runs on the host (not in Docker). Reached via `host.docker.internal:11434`.
- **OpenWebUI** is a chat UI that talks to the ingress via the `/v1/chat/completions` shim.
- **PipelineRunner** is the core orchestrator in `coretex/runtime/pipeline.py`. It accesses components through the module registry — never imports them directly.

---

## Invariants (Architectural Constraints)

1. Single orchestration entry point: `POST /ingest`
2. Exactly **two LLM calls** per successful (non-ambiguous) request:
   - Call 1: Classifier → intent classification
   - Call 2: Worker → JSON action envelope generation
3. Router is **pure Python** — no LLM calls, no probabilistic decisions
4. **No memory layer** — each request is stateless
5. **Agents never execute tools directly** — only `ToolExecutor` can run tools
6. **No autonomous planning loops**
7. Worker prompts use string concatenation for user input — never `str.format()` or f-strings
8. **Runtime (`coretex/`) never imports from `modules/`** — coupling is only through registry lookups

---

## File Structure

```
coretex/
  __init__.py
  interfaces/
    __init__.py
    classifier.py       — Classifier ABC + ClassificationResult(intent, confidence, source) dataclass
    router.py           — Router ABC
    worker.py           — Worker ABC
    model_provider.py   — ModelProvider ABC
  registry/
    __init__.py
    tool_registry.py    — Tool dataclass, ToolRegistry (raises ValueError on dup/unknown, logs registry_lookup_failed)
    module_registry.py  — ModuleRegistry: classifier/router/worker (raises ValueError on dup/unknown, logs registry_lookup_failed)
    model_registry.py   — ModelProviderRegistry (raises ValueError on dup/unknown, logs registry_lookup_failed)
    pipeline_registry.py — PipelineRegistry (raises ValueError on dup/unknown, logs registry_lookup_failed)
  runtime/
    __init__.py
    context.py          — ExecutionContext dataclass (request_id, input, intent, confidence, handler, t_start, timestamp, metadata)
    events.py           — EventBus (emit/emit_warning/emit_error structured log wrappers)
    loader.py           — ModuleLoader (signature validation, empty-registration warning, load_all() lifecycle events)
    executor.py         — AgentAction, ToolExecutor, parse_agent_output
    pipeline.py         — PipelineRunner (full log lifecycle, explicit failure categories)
  config/
    __init__.py
    settings.py         — Pydantic-settings config (debug_router, log_level, etc.)

modules/
  __init__.py
  classifier_basic/
    __init__.py
    classifier.py       — ClassifierBasic(Classifier): prefix checks + Ollama LLM call
    module.py           — register(module_registry, tool_registry, model_registry)
  router_simple/
    __init__.py
    router.py           — RouterSimple(Router): ROUTES dict + route() + debug_router logging
    module.py
  worker_llm/
    __init__.py
    worker.py           — WorkerLLM(Worker): _PROMPTS dict + Ollama LLM call
    module.py
  tools_filesystem/
    __init__.py
    filesystem.py       — read_file(path) function
    module.py
  model_provider_ollama/
    __init__.py
    provider.py         — OllamaProvider(ModelProvider)
    module.py

distributions/
  __init__.py
  cortx/
    __init__.py
    main.py             — FastAPI app: /ingest, /v1/chat/completions, /debug/routes, /health, /v1/models
    models.py           — Pydantic schemas: IngestRequest, IngestResponse
    bootstrap.py        — Creates registries, loads all modules via loader.load_all()

tests/
  test_smoke.py         — All tests (106 unit + integration via TestClient, Ollama fully mocked)

docs/
  runtime.md            — Runtime architecture, pipeline execution, failure behaviour
  module_development.md — How to build modules
  distributions.md      — How to build distributions

Dockerfile              — python:3.11-slim, uvicorn distributions.cortx.main:app
docker-compose.yml      — ingress + openwebui services on isolated bridge network
requirements.txt        — fastapi, uvicorn, httpx, pydantic, pydantic-settings, pytest
pytest.ini              — pythonpath = . so all package imports resolve without install
```

---

## Module Detail

### `coretex/config/settings.py`

Pydantic `BaseSettings` subclass. All values overridable via environment variables or `.env` file.

| Setting               | Default                              | Purpose                                        |
|-----------------------|--------------------------------------|------------------------------------------------|
| `ollama_base_url`     | `http://host.docker.internal:11434`  | Base URL for all Ollama API calls              |
| `classifier_model`    | `llama3.2:3b`                        | Model used by the classifier agent             |
| `worker_model`        | `llama3.2:3b`                        | Model used by the worker agent                 |
| `classifier_timeout`  | `60` (seconds)                       | httpx timeout for classifier call              |
| `worker_timeout`      | `300` (seconds)                      | httpx timeout for worker call                  |
| `max_tokens`          | `256`                                | `num_predict` passed to worker                 |
| `log_level`           | `INFO`                               | Python logging level                           |
| `debug_router`        | `false`                              | When `true`, emit event=router_decision DEBUG  |

Singleton `settings` imported by all modules that need configuration.

---

### `coretex/interfaces/`

ABCs establishing contracts:

- **`Classifier`** — `async classify(input: str, request_id: str = "") -> ClassificationResult`
- **`Router`** — `route(intent: str, request_id: str = "", **kwargs) -> str`
- **`Worker`** — `async generate(input: str, intent: str, request_id: str = "") -> str`
- **`ModelProvider`** — `async generate(model, prompt, **kwargs) -> str`, `async chat(model, messages, **kwargs) -> str`
- **`ClassificationResult`** — `@dataclass` with `intent: str, confidence: float, source: str`

---

### `coretex/registry/`

All four registries follow identical safety rules:
- `register()` raises `ValueError("Component already registered: <name>")` on duplicates
- `get()` raises `ValueError("Unknown component: <name>")` on unknown name AND logs `event=registry_lookup_failed component=<type> name=<name>`

- **`ToolRegistry`** — stores `Tool` dataclasses. `register(name, desc, schema, fn)`, `get(name)`, `list()`.
- **`Tool`** — `@dataclass` with `name`, `description`, `input_schema`, `function`. `execute(args, request_id)` logs `event=tool_execute` / `event=tool_execute_complete`.
- **`ModuleRegistry`** — stores classifier/router/worker instances by name. `register_classifier/router/worker()`, `get_classifier/router/worker()`, `mark_loaded()`, `list_loaded()`.
- **`ModelProviderRegistry`** — stores `ModelProvider` instances. `register(name, provider)`, `get(name)`, `list()`.
- **`PipelineRegistry`** — placeholder for v0.4.0 configurable pipelines. `register(name, pipeline)`, `get(name)`, `list()`.

---

### `coretex/runtime/context.py`

```python
@dataclass
class ExecutionContext:
    user_input: str
    request_id: str                    # auto-generated UUID hex
    intent: Optional[str]              # set after classification
    confidence: float                  # set after classification
    handler: Optional[str]             # set after routing
    t_start: float                     # monotonic timestamp (for latency)
    timestamp: float                   # wall-clock time.time() (for observability)
    metadata: Optional[Dict[str, Any]] # optional free-form module metadata
```

---

### `coretex/runtime/loader.py`

`ModuleLoader` loads modules at startup. Validation steps per `load()` call:

1. `importlib.import_module(path)` — raises `ImportError` on failure (logs `event=module_import_failed`)
2. Check `mod.register` exists and is callable — raises `ValueError("Module '...' has no register() function")`
3. Check signature has all three params (`module_registry`, `tool_registry`, `model_registry`) — raises `ValueError("Invalid module register() signature ...")`
4. Execute `mod.register(...)`, count new registrations
5. If 0 components registered: `logger.warning("event=module_loaded ... warning=module_registered_nothing")`
6. Otherwise: `logger.info("event=module_loaded module=... registered_components=N")`

`load_all(paths)` emits `event=module_loading_start` and `event=module_loading_complete`.

---

### `coretex/runtime/executor.py`

- **`AgentAction`** — typed wrapper: `action`, `tool`, `args`, `content`. `from_dict()` is the constructor.
- **`ToolExecutor`** — dispatches on `action.action`: `"respond"` returns `action.content`; `"tool"` looks up and calls the tool; unknown action raises `ValueError`.
- **`parse_agent_output(raw)`** — parses JSON string into `AgentAction`. Logs `event=agent_output_received`. Raises and logs `event=agent_output_parse_error` on failure.

---

### `coretex/runtime/pipeline.py`

`PipelineRunner.run(context)` returns `(response_text, intent, confidence)`.

Full log lifecycle:
```
event=request_received
event=classifier_start
event=classifier_complete    (includes intent, confidence, duration_ms)
event=router_selected        (includes intent, handler)
event=worker_start           (includes worker name, intent)
event=worker_complete        (includes duration_ms)
event=agent_output_received
event=tool_execute / event=tool_execute_complete
event=request_complete       (includes all latencies: classifier_latency_ms, worker_latency_ms, total_latency_ms)
```

Failure categories:

| Failure | Event | Behaviour |
|---------|-------|-----------|
| Classifier HTTP error | `event=pipeline_classifier_failure` | `intent=ambiguous`, clarification response |
| Worker HTTP error | `event=pipeline_worker_failure` | worker failure response, `intent=ambiguous` |
| Tool lookup/runtime error | `event=pipeline_tool_failure` | worker failure response |
| Agent JSON parse error | `event=pipeline_agent_parse_failure` | raw text treated as direct response |

---

### `modules/classifier_basic/classifier.py`

**Stage 1 — Deterministic prefix check (no LLM)**

Lowercases input and matches against prefix lists:
- `_EXECUTION_PREFIXES` → `intent="execution", confidence=0.95`
- `_PLANNING_PREFIXES` → `intent="planning", confidence=0.95`
- `_AMBIGUOUS_SHORT` → `intent="ambiguous", confidence=0.95`

**Stage 2 — LLM call via `_call_ollama()`**

Calls Ollama `/api/chat`. Up to 2 attempts. Falls back to `ClassificationResult(intent="ambiguous", confidence=0.0)` after 2 failures.

**`_parse()` normalisation:** strip markdown fences → `json.loads` → `_ClassifierResponse` Pydantic validation → field-name scan → alias map → graceful `ambiguous` fallback.

`_ClassifierResponse` enforces `Literal["execution", "planning", "analysis", "ambiguous"]` on the intent field.

---

### `modules/router_simple/router.py`

```python
ROUTES = {
    "execution": "worker",
    "planning":  "worker",
    "analysis":  "worker",
    "ambiguous": "clarify",
}
```

`RouterSimple.route(intent)` returns handler name. Unknown intents → `"clarify"`. When `settings.debug_router == True`, emits `event=router_decision` at DEBUG level.

---

### `modules/worker_llm/worker.py`

Intent-aware via `_PROMPTS` dict (execution, planning, analysis, `_FALLBACK_PROMPT`). All prompts instruct the LLM to return a JSON action envelope. User input is appended with `+` — never `str.format()`.

---

### `modules/tools_filesystem/filesystem.py`

`read_file(path: str) -> str` — reads file text. Returns `"File not found: <path>"` on missing file (never raises).

---

### `distributions/cortx/bootstrap.py`

Creates three registry singletons. Calls `ModuleLoader.load_all([...])` with all five module paths. Emits module loading lifecycle events. Imported by `main.py` at module load time.

---

### `distributions/cortx/main.py`

FastAPI endpoints:
- `POST /ingest` — creates `ExecutionContext`, calls `pipeline.run()`, returns `IngestResponse`
- `POST /v1/chat/completions` — OpenAI-compatible shim, extracts last user message, calls pipeline
- `GET /debug/routes` — returns `ROUTES` dict
- `GET /health` — returns `{"status": "ok"}`
- `GET /v1/models` — returns the `agentic` model descriptor for OpenWebUI

---

### `distributions/cortx/models.py`

- **`IngestRequest`** — `input: str`. Validator rejects empty/whitespace (HTTP 422).
- **`IngestResponse`** — `intent: str`, `confidence: float`, `response: str`.

---

## Testing

All tests in `tests/test_smoke.py`. **106 tests** covering all components.

**Test runner:** `pytest tests/test_smoke.py -v`

Async tests use `@pytest.mark.anyio`. All Ollama calls mocked with `AsyncMock`.

Test categories:
- Router unit tests (pure Python)
- Classifier internal validation and parsing
- Classifier behaviour with mocked Ollama
- `/ingest` happy path and edge cases
- `/v1/chat/completions` shim
- Health and model list endpoints
- Worker prompt content validation
- Tool registry: register, get, duplicate, unknown, list
- AgentAction: parsing, defaults
- ToolExecutor: respond, tool, unknown action, missing tool name, runtime exception
- `parse_agent_output`: valid, invalid JSON
- Filesystem tool
- Registry duplicate and unknown-lookup tests (all four registries)
- ModuleLoader: valid module, missing register(), wrong signature, empty registration, ImportError, load_all() lifecycle
- Pipeline failure scenarios: classifier HTTP failure, worker HTTP failure, invalid JSON, tool lookup failure, tool runtime exception
- Logging event tests: all key pipeline events present, latency fields present
- ExecutionContext: timestamp, metadata
- Router debug_router setting

---

## What Has Been Built

- **v0.1.x**: Skeleton FastAPI service, `/ingest`, classifier LLM call, deterministic router, basic worker LLM call, smoke tests, Docker/Compose, OpenWebUI integration.
- **v0.2.x**: Intent-aware worker prompts, classifier hardening (prefix checks, alias normalisation, markdown fence stripping, retry-with-fallback), graceful failure handling, request correlation IDs, structured logging, `DEBUG_ROUTER`, `GET /debug/routes`, tool execution layer, `read_file` tool, worker JSON action envelope prompts, integrated executor into pipeline.
- **v0.3.0**: Runtime extraction — `coretex/` package (interfaces, registries, executor, pipeline, loader, context, events, config); `modules/`; `distributions/cortx/`; updated Dockerfile; removed legacy `app/`, `core/`, `tools/`.
- **v0.3.x Stabilisation**: Hardened pipeline with explicit failure categories and full log lifecycle; standardised registry validation (consistent error messages, `event=registry_lookup_failed`); ModuleLoader signature validation, empty-registration warning, `load_all()` lifecycle events; `ExecutionContext` metadata and timestamp fields; router `debug_router` logging; expanded test suite (64 → 106 tests); `docs/runtime.md`, `docs/module_development.md`, `docs/distributions.md`.

---

## What Has NOT Been Built (Non-Goals)

- Memory / conversation history
- Streaming responses
- Multi-agent coordination
- Task Graph / Planner orchestration
- Authentication / authorisation
- Persistent storage of any kind
- Rate limiting
- Multiple model backends (only Ollama supported)
- Async tool execution
- Plugin dependency graphs
