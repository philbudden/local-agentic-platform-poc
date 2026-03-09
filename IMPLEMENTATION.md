# Implementation Description — COREtex Runtime Platform

> This document describes the current state of the implementation as of v0.3.0 (Runtime Extraction).
> It is intended to be read as context by an LLM when planning future phases of the project.

---

## Purpose

COREtex is a **local agentic runtime platform** that routes user requests through a two-stage LLM pipeline (classifier → worker) using locally-hosted models via [Ollama](https://ollama.com), then passes the agent's JSON output through a deterministic tool execution layer.

v0.3.0 restructures the codebase from a monolithic `app/` package into a **modular runtime** with three distinct layers: the `coretex/` runtime package (interfaces, registries, execution engine), `modules/` (pluggable component implementations), and `distributions/` (assembled applications).

---

## High-Level Architecture

```
User (browser)
  └─► OpenWebUI  (port 3000)
        └─► POST /v1/chat/completions  ← OpenAI-compatible shim
              └─► POST /ingest         ← distribution entry point (cortx_local)
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
    classifier.py       — Classifier ABC + ClassificationResult(intent, confidence) dataclass
    router.py           — Router ABC
    worker.py           — Worker ABC
    model_provider.py   — ModelProvider ABC
  registry/
    __init__.py
    tool_registry.py    — Tool dataclass, ToolRegistry
    module_registry.py  — ModuleRegistry (stores classifier/router/worker instances by name)
    model_registry.py   — ModelProviderRegistry
    pipeline_registry.py — PipelineRegistry
  runtime/
    __init__.py
    context.py          — ExecutionContext dataclass (request_id, input, metadata)
    events.py           — EventBus
    loader.py           — ModuleLoader (calls register() on each module)
    executor.py         — AgentAction, ToolExecutor, parse_agent_output
    pipeline.py         — PipelineRunner
  config/
    __init__.py
    settings.py         — Pydantic-settings config

modules/
  __init__.py
  classifier_basic/
    __init__.py
    classifier.py       — ClassifierBasic(Classifier): prefix checks + Ollama LLM call
    module.py           — register(module_registry, tool_registry, model_registry)
  router_simple/
    __init__.py
    router.py           — RouterSimple(Router): ROUTES dict + route() function
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
  cortx_local/
    __init__.py
    main.py             — FastAPI app: /ingest, /v1/chat/completions, /debug/routes, /health
    models.py           — Pydantic schemas: IngestRequest, IngestResponse
    bootstrap.py        — Creates registries, loads all modules via ModuleLoader

tests/
  test_smoke.py         — All tests (unit + integration via TestClient, Ollama fully mocked)

Dockerfile              — python:3.11-slim, uvicorn distributions.cortx_local.main:app
docker-compose.yml      — ingress + openwebui services on an isolated bridge network
requirements.txt        — fastapi, uvicorn, httpx, pydantic, pydantic-settings, pytest
pytest.ini              — pythonpath = . so all package imports resolve without install
```

---

## Module Detail

### `coretex/config/settings.py`

Pydantic `BaseSettings` subclass. All values overridable via environment variables or `.env` file.

| Setting               | Default                              | Purpose                              |
|-----------------------|--------------------------------------|--------------------------------------|
| `ollama_base_url`     | `http://host.docker.internal:11434`  | Base URL for all Ollama API calls    |
| `classifier_model`    | `llama3.2:3b`                        | Model used by the classifier agent   |
| `worker_model`        | `llama3.2:3b`                        | Model used by the worker agent       |
| `classifier_timeout`  | `60` (seconds)                       | httpx timeout for classifier call    |
| `worker_timeout`      | `300` (seconds)                      | httpx timeout for worker call        |
| `max_tokens`          | `256`                                | `num_predict` passed to worker       |
| `log_level`           | `INFO`                               | Python logging level                 |
| `debug_router`        | `false`                              | When `true`, log prompts at DEBUG    |

Singleton `settings` imported by all modules.

---

### `coretex/interfaces/`

ABCs establishing contracts:

- **`Classifier`** — `classify(input: str, **kwargs) -> ClassificationResult`
- **`Router`** — `route(intent: str) -> str`
- **`Worker`** — `generate(input: str, intent: str, **kwargs) -> str`
- **`ModelProvider`** — `complete(prompt: str, **kwargs) -> str`, `is_available() -> bool`
- **`ClassificationResult`** — `@dataclass` with `intent: str, confidence: float`

---

### `coretex/registry/`

- **`ToolRegistry`** — stores `Tool` dataclasses by name. `register()` raises `ValueError` on duplicates. `get()` raises `ValueError` for unknown tools (logs `event=tool_lookup_failed`). `list()` returns all names.
- **`Tool`** — `@dataclass` with `name`, `description`, `input_schema`, `function`. `execute(args)` calls the function and logs `event=tool_execute` / `event=tool_execute_complete`.
- **`ModuleRegistry`** — stores component instances by name. `register_classifier(name, instance)`, `get_classifier(name)`, and equivalents for router, worker.

---

### `coretex/runtime/executor.py`

- **`AgentAction`** — typed wrapper for agent JSON output: `action`, `tool`, `args`, `content`. `from_dict()` is the primary constructor.
- **`ToolExecutor`** — dispatches on `action.action`: `"respond"` returns `action.content` directly; `"tool"` looks up the tool and calls it; anything else raises `ValueError`.
- **`parse_agent_output(raw)`** — parses a JSON string into `AgentAction`. Logs `event=agent_output_received`. Raises and logs `event=agent_output_parse_error` on failure.

---

### `coretex/runtime/pipeline.py`

`PipelineRunner` wraps all three registries. `run(context: ExecutionContext) -> Tuple[str, str, float]` returns `(response_text, intent, confidence)`.

Pipeline:
1. Get classifier from module_registry → `ClassificationResult`
2. Get router from module_registry → handler name
3. If handler is `"clarify"`: return `_CLARIFY_RESPONSE` directly
4. If handler is `"worker"`: get worker → raw JSON string → `parse_agent_output` → `ToolExecutor.execute` → response text
5. On JSON parse error: treat raw as direct response (graceful fallback)
6. On tool `ValueError`: return `_WORKER_FAILURE_RESPONSE`
7. On `httpx.HTTPError`: return `_WORKER_FAILURE_RESPONSE` with `intent="ambiguous", confidence=0.0`

---

### `coretex/runtime/loader.py`

`ModuleLoader.load()` imports each module's `module.py` and calls `register(module_registry=..., tool_registry=..., model_registry=...)`. Called once at startup from `distributions/cortx_local/bootstrap.py`.

---

### `modules/classifier_basic/classifier.py`

**Stage 1 — Deterministic prefix check (no LLM)**

Lowercases the input and matches against:
- `_EXECUTION_PREFIXES`: `("write", "generate", "create", "compose", "draft", "produce", "summarise", "summarize", "translate", "calculate", "code", "list")` → `intent="execution", confidence=0.95`
- `_PLANNING_PREFIXES`: `("how do i", "how would i", "how can i", "what steps")` → `intent="planning", confidence=0.95`
- `_AMBIGUOUS_SHORT`: `("help", "hi", "hello", "hey", "ok", "okay", "thanks")` → `intent="ambiguous", confidence=0.95`

**Stage 2 — LLM call via `_call_ollama()`**

Calls Ollama `/api/chat`. Up to 2 attempts. Falls back to `ClassificationResult(intent="ambiguous", confidence=0.0)` after 2 failures.

**`_parse()` normalisation:** strip markdown fences → `json.loads` → `_ClassifierResponse` Pydantic validation → field-name scan → alias map → graceful `ambiguous` fallback.

`_ClassifierResponse` is an internal Pydantic model (prefixed `_`) that enforces `Literal["execution", "planning", "analysis", "ambiguous"]` on the intent field.

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

`RouterSimple.route(intent)` returns the handler name. Unknown intents fall back to `"clarify"`.

---

### `modules/worker_llm/worker.py`

Generates the final user-facing response. Intent-aware via `_PROMPTS` dict (execution, planning, analysis, plus a `_FALLBACK_PROMPT`). All prompts instruct the LLM to return:
```json
{"action": "respond", "content": "..."}
```
or a tool call:
```json
{"action": "tool", "tool": "<name>", "args": {"<key>": "<value>"}}
```

User input is **appended** with `+` — never `str.format()`. `generate()` raises `httpx.HTTPError` on failure; the caller handles gracefully.

---

### `modules/tools_filesystem/filesystem.py`

`read_file(path: str) -> str` — reads and returns the text content of a file. Returns `"File not found: <path>"` if the file does not exist (never raises).

---

### `modules/model_provider_ollama/provider.py`

`OllamaProvider(ModelProvider)` — wraps the Ollama HTTP API. `complete(prompt, model, timeout, **kwargs)` calls `/api/generate`. `is_available()` checks connectivity.

---

### `distributions/cortx_local/bootstrap.py`

Creates `module_registry`, `tool_registry`, `model_registry` singletons. Calls `ModuleLoader.load()` for all five modules. Imported by `main.py` at module load time.

---

### `distributions/cortx_local/main.py`

The FastAPI application.

#### `POST /ingest`

1. Generate `request_id = uuid4().hex`; log `event=request_received`
2. Create `ExecutionContext(request_id=request_id, input=request.input)`
3. Call `pipeline.run(context)` → `(response_text, intent, confidence)`
4. Log `event=request_complete` with timing
5. Return `IngestResponse`

**Other endpoints:**
- `POST /v1/chat/completions` — OpenAI-compatible shim, extracts last user message and calls `/ingest` internally
- `GET /debug/routes` — returns `ROUTES` dict from `modules.router_simple.router`
- `GET /health` — returns `{"status": "ok"}`
- `GET /v1/models` — returns the agentic model descriptor for OpenWebUI

---

### `distributions/cortx_local/models.py`

Two Pydantic v2 models:
- **`IngestRequest`** — `input: str`. A `field_validator` rejects empty or whitespace-only strings (returns HTTP 422).
- **`IngestResponse`** — `intent: str`, `confidence: float`, `response: str`.

---

## Testing

All tests are in `tests/test_smoke.py`. **64 tests** covering all components.

**Test runner:** `pytest tests/test_smoke.py -v`

Async tests use `@pytest.mark.anyio`. All Ollama calls are mocked with `AsyncMock`. Patch targets use the new module paths:
- `modules.classifier_basic.classifier.ClassifierBasic.classify`
- `modules.worker_llm.worker.WorkerLLM.generate`

Mock fixtures return `ClassificationResult(intent=..., confidence=...)` from `coretex.interfaces.classifier`.

---

## Infrastructure

### Dockerfile

- Base image: `python:3.11-slim`
- Workdir: `/workspace`
- Copies: `requirements.txt`, `coretex/`, `modules/`, `distributions/`
- Entry point: `uvicorn distributions.cortx_local.main:app --host 0.0.0.0 --port 8000`

### `docker-compose.yml`

Two services on an isolated `agentic` bridge network: `ingress` (port 8000) and `openwebui` (port 3000). OpenWebUI has `ENABLE_OLLAMA_API=false`.

---

## What Has Been Done

- **v0.1.x**: Skeleton FastAPI service, `/ingest`, classifier LLM call, deterministic router, basic worker LLM call, smoke tests, Docker/Compose setup, OpenWebUI integration.
- **v0.2.x**: Intent-aware worker prompts, classifier hardening (prefix checks, alias normalisation, field-name scanning, markdown fence stripping, retry-with-fallback), graceful worker failure handling, request correlation IDs, structured log events, `DEBUG_ROUTER`, `GET /debug/routes`, tool execution layer, `read_file` tool, `bootstrap_tools.py`, updated worker prompts for JSON action envelopes, integrated executor into pipeline.
- **v0.3.0**: Runtime extraction — `coretex/` package (interfaces, registries, executor, pipeline, loader, context, events, config); `modules/` (classifier_basic, router_simple, worker_llm, tools_filesystem, model_provider_ollama); `distributions/cortx_local/` (FastAPI app, models, bootstrap); updated Dockerfile; removed legacy `app/`, `core/`, `tools/`, `bootstrap_tools.py`; updated test suite for new import paths.

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
- Separate classifier and worker model instances (both default to the same model)
