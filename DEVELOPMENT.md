# Development Guide

This guide is written for developers who want to extend or modify COREtex. It covers the v0.3.x architecture, core components, versioning conventions, and how to add new modules, tools, routes, and API endpoints.

---

## Architecture overview

COREtex v0.3.x is a **runtime platform** composed of three distinct layers. The key architectural rule is: **the runtime never imports from modules**. All coupling goes through interfaces and registries.

```
coretex/              ← Runtime platform (never imports from modules/)
  runtime/          ← PipelineRunner, ToolExecutor, ModuleLoader, ExecutionContext, EventBus
  interfaces/       ← ABCs: Classifier, Router, Worker, ModelProvider
  registry/         ← ToolRegistry, ModuleRegistry, ModelProviderRegistry, PipelineRegistry
  config/           ← Settings (Pydantic BaseSettings)

modules/            ← Implementations registered at startup
  classifier_basic/        ← Intent classifier (prefix checks + LLM)
  router_simple/           ← Deterministic dict-based router
  worker_llm/              ← LLM response generator with intent-aware prompts
  tools_filesystem/        ← read_file tool
  model_provider_ollama/   ← Ollama inference backend

distributions/
  cortx/      ← FastAPI ingress + OpenWebUI (main.py, bootstrap.py, models.py)

docs/               ← Extended documentation
  runtime.md              ← Runtime internals, pipeline flow, failure catalogue
  module_development.md   ← Module authoring guide
  distributions.md        ← Distribution system and bootstrap pattern
```

### Request flow

```
User input
    │
    ▼
POST /ingest  (distributions/cortx/main.py)
    │  Creates ExecutionContext(request_id=uuid, input=..., timestamp=time.time())
    │
    ▼
PipelineRunner.run(context)  (coretex/runtime/pipeline.py)
    │
    ├── classifier_start log
    ├── module_registry.get_classifier("classifier_basic")
    │     ClassifierBasic.classify(input) → ClassificationResult(intent, confidence)
    ├── classifier_complete log (with duration_ms)
    │
    ├── router_selected log
    ├── module_registry.get_router("router_simple")
    │     RouterSimple.route(intent) → handler str
    │
    ├── [if handler == "clarify"] → return clarification response
    │
    ├── worker_start log
    └── module_registry.get_worker("worker_llm")
          WorkerLLM.generate(input, intent) → raw JSON string
          worker_complete log (with duration_ms)
              │
              ▼
         parse_agent_output(raw) → AgentAction
              │
              ▼
         ToolExecutor.execute(action) → result string
              │
         ┌────┴─────┐
         │           │
     "respond"    "tool"
         │           │
     return         tool_registry.get(name).execute(args)
     content
    │
    ▼
request_complete log (total_latency_ms, classifier_latency_ms, handler)
```

**Key design rules:**
- The runtime (`coretex/`) never imports from `modules/`. Access is through registry lookups.
- Agents (LLMs) never execute tools directly — only `ToolExecutor` can.
- The router is pure Python — no LLM calls, no probabilistic decisions.
- Each request is stateless — no memory, no conversation history.
- Exactly 2 LLM calls per non-ambiguous request (classifier + worker).

---

## File structure

```
coretex/
  __init__.py
  interfaces/
    __init__.py
    classifier.py       — Classifier ABC + ClassificationResult dataclass
    router.py           — Router ABC
    worker.py           — Worker ABC
    model_provider.py   — ModelProvider ABC
  registry/
    __init__.py
    tool_registry.py    — Tool dataclass, ToolRegistry
    module_registry.py  — ModuleRegistry (holds classifier/router/worker instances)
    model_registry.py   — ModelProviderRegistry
    pipeline_registry.py — PipelineRegistry
  runtime/
    __init__.py
    context.py          — ExecutionContext dataclass (request_id, input, timestamp, metadata)
    events.py           — EventBus
    loader.py           — ModuleLoader (with signature validation, load_all())
    executor.py         — AgentAction, ToolExecutor, parse_agent_output
    pipeline.py         — PipelineRunner with full log lifecycle and failure handling
  config/
    __init__.py
    settings.py         — Pydantic-settings config (Settings singleton)

modules/
  __init__.py
  classifier_basic/
    __init__.py
    classifier.py       — ClassifierBasic(Classifier)
    module.py           — register() entry point
  router_simple/
    __init__.py
    router.py           — RouterSimple(Router), ROUTES dict, debug_router support
    module.py
  worker_llm/
    __init__.py
    worker.py           — WorkerLLM(Worker), _PROMPTS dict
    module.py
  tools_filesystem/
    __init__.py
    filesystem.py       — read_file function
    module.py
  model_provider_ollama/
    __init__.py
    provider.py         — OllamaProvider(ModelProvider)
    module.py

distributions/
  __init__.py
  cortx/
    __init__.py
    main.py             — FastAPI app entry point
    models.py           — Pydantic schemas: IngestRequest, IngestResponse
    bootstrap.py        — Creates registries, calls ModuleLoader.load_all()

docs/
  runtime.md            — Pipeline internals, failure catalogue, log event reference
  module_development.md — Module authoring, register() contract, common errors
  distributions.md      — Distribution system, bootstrap pattern, Docker deployment

tests/
  test_smoke.py         — Full test suite: 106 tests (unit + integration, no Ollama required)

Dockerfile              — python:3.11-slim, runs uvicorn distributions.cortx.main:app
docker-compose.yml      — ingress + openwebui on an isolated bridge network
requirements.txt        — All Python dependencies
pytest.ini              — Sets pythonpath = . so imports resolve without install
```

---

## Core components

### `coretex/interfaces/`

Abstract base classes defining what each component type must implement:

- **`Classifier`** — `classify(input: str, **kwargs) -> ClassificationResult`. `ClassificationResult` is a dataclass: `intent: str, confidence: float`.
- **`Router`** — `route(intent: str) -> str`.
- **`Worker`** — `generate(input: str, intent: str, **kwargs) -> str`.
- **`ModelProvider`** — `complete(prompt: str, **kwargs) -> str`, `is_available() -> bool`.

### `coretex/registry/`

All four registries follow the same pattern: `register()`, `get()`, `list()`. Duplicate names raise `ValueError("Component already registered: <name>")`. Unknown names raise `ValueError("Unknown component: <name>")` and log `event=registry_lookup_failed`.

- **`ToolRegistry`** — stores `Tool` dataclasses by name.
- **`ModuleRegistry`** — stores classifier/router/worker instances by name.
- **`ModelProviderRegistry`** — stores `ModelProvider` instances by name.
- **`PipelineRegistry`** — stores pipeline instances by name.

### `coretex/runtime/context.py`

`ExecutionContext` dataclass fields:
- `request_id: str` — UUID for the request, threaded through all log events.
- `input: str` — the user's raw input text.
- `timestamp: float` — wall-clock time at context creation (`time.time()`).
- `metadata: Optional[Dict[str, Any]]` — optional caller-supplied data (defaults to `None`).

### `coretex/runtime/executor.py`

- **`AgentAction`** — typed representation of the agent's JSON output (action, tool, args, content). `from_dict()` factory with defaults.
- **`ToolExecutor`** — dispatches on `action.action`: `"respond"` returns content; `"tool"` calls `ToolRegistry.get(name).execute(args)`.
- **`parse_agent_output(raw)`** — parses raw JSON string → `AgentAction`. Raises `json.JSONDecodeError` on invalid input.

### `coretex/runtime/pipeline.py`

`PipelineRunner.run(context: ExecutionContext) -> Tuple[str, str, float]` — the core orchestrator. Returns `(response_text, intent, confidence)`. Failure categories:
- `pipeline_classifier_failure` — classifier HTTP error; returns clarify response with `intent=ambiguous`.
- `pipeline_worker_failure` — worker HTTP error; returns failure response.
- `pipeline_agent_parse_failure` — invalid JSON from worker; treats as plain text.
- `pipeline_tool_failure` — unknown tool name; returns failure response.
- `pipeline_worker_failure` (runtime) — tool/executor exception; returns failure response.

### `coretex/runtime/loader.py`

`ModuleLoader.load(module_path)` — imports the module and calls `register(module_registry, tool_registry, model_registry)`.

`ModuleLoader.load_all(paths)` — wraps multiple `load()` calls with `module_loading_start` / `module_loading_complete` lifecycle events.

Validation:
- `register()` must accept `module_registry`, `tool_registry`, `model_registry` parameters — raises `ValueError("Invalid module register() signature ...")` otherwise.
- Warns with `event=module_loaded ... warning=module_registered_nothing` when 0 components are registered.

---

## Versioning conventions

COREtex follows strict semantic versioning. From `documentation/AGENTS.md`:

- All commits must be on a feature branch, never directly on `main`.
- Branch names follow `feature/v<X>.<Y>-<description>`.
- Commits are single units of work with the format:
  ```
  <type>(<scope>): <description> v<X>.<Y>.<Z>
  ```
  Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.
- Version increments are sequential: each commit advances the patch number by 1.
- Co-author every commit with `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

Example:
```
feat(runtime): add timestamp field to ExecutionContext v0.3.16
```

---

## Logging conventions

All log lines use `event=<name>` as the first field, followed by key-value pairs:

```python
logger.info("event=tool_execute tool=%s request_id=%s", name, request_id)
```

Standard log events in sequence for a tool call request:

```
event=request_received        request_id=<id>
event=classifier_start        request_id=<id> classifier=<name>
event=classifier_complete     request_id=<id> intent=<intent> confidence=<float> duration_ms=<int>
event=router_selected         request_id=<id> intent=<intent> handler=<handler>
event=worker_start            request_id=<id> worker=<name> intent=<intent>
event=worker_complete         request_id=<id> duration_ms=<int>
event=tool_execute            request_id=<id> tool=<name>
event=tool_execute_complete   request_id=<id> tool=<name>
event=request_complete        request_id=<id> intent=<intent> confidence=<float> handler=<handler> total_latency_ms=<int>
```

Set `DEBUG_ROUTER=true` to additionally log `event=router_decision` at DEBUG level.

---

## How to add a new tool

### 1. Add the function to the appropriate module (or create a new one)

```python
# modules/tools_filesystem/filesystem.py

def write_file(path: str, content: str) -> str:
    """Write content to a file at path. Returns an error string on failure."""
    try:
        pathlib.Path(path).write_text(content)
        return f"Written to {path}"
    except OSError as exc:
        return f"Write failed: {exc}"
```

Tools must:
- Accept only keyword arguments (the executor calls `function(**args)`).
- Return a string or serialisable value.
- Handle their own errors gracefully — return an error string rather than raising.

### 2. Register in the module's `module.py`

```python
def register(module_registry, tool_registry, model_registry):
    from modules.tools_filesystem.filesystem import write_file
    tool_registry.register(
        name="write_file",
        description="Write text content to a local file",
        input_schema={"path": "string", "content": "string"},
        function=write_file,
    )
```

### 3. Update the worker prompt

To make the LLM aware of the new tool, add a description to the relevant prompt template in `modules/worker_llm/worker.py`.

### 4. Add tests

```python
def test_write_file_tool():
    from modules.tools_filesystem.filesystem import write_file
    import tempfile, pathlib
    f = pathlib.Path(tempfile.mktemp())
    result = write_file(path=str(f), content="hello")
    assert "Written to" in result

def test_bootstrap_registers_write_file():
    from distributions.cortx.bootstrap import tool_registry
    assert "write_file" in tool_registry.list()
```

---

## How to add a new module

Each module lives in `modules/<name>/` with a `module.py` containing a `register()` function:

```python
# modules/my_module/module.py

def register(module_registry, tool_registry, model_registry):
    from modules.my_module.my_class import MyClassifier
    module_registry.register_classifier("my_classifier", MyClassifier())
```

`register()` **must** accept `module_registry`, `tool_registry`, and `model_registry` as parameters — `ModuleLoader` validates the signature before calling it.

Then add the module path to `bootstrap.py`:

```python
loader.load_all([
    "modules.classifier_basic.module",
    "modules.router_simple.module",
    "modules.worker_llm.module",
    "modules.tools_filesystem.module",
    "modules.model_provider_ollama.module",
    "modules.my_module.module",   # ← new
])
```

See [`docs/module_development.md`](docs/module_development.md) for the complete authoring guide.

---

## How to add a new intent route

### 1. Add prefix checks in `modules/classifier_basic/classifier.py`

```python
_RETRIEVAL_PREFIXES = ("find ", "search for ", "look up ")
```

Add a check in `classify()` to return early for this prefix.

### 2. Update `_INTENT_ALIASES` in the classifier

```python
_INTENT_ALIASES = {
    ...
    "retrieve": "retrieval",
    "search": "retrieval",
}
```

### 3. Update `ROUTES` in `modules/router_simple/router.py`

```python
ROUTES: dict[str, str] = {
    ...
    "retrieval": "worker",
}
```

### 4. Add a prompt template in `modules/worker_llm/worker.py`

```python
_PROMPTS: dict[str, str] = {
    ...
    "retrieval": (
        "You are a retrieval assistant. Find the most relevant information.\n"
        'You MUST respond with valid JSON: {"action": "respond", "content": "..."}\n\n'
        "User request: "
    ),
}
```

---

## How to add a new API endpoint

Add a route to `distributions/cortx/main.py`:

```python
@app.get("/my-endpoint")
async def my_endpoint() -> dict:
    """One-line docstring."""
    return {"key": "value"}
```

All endpoints should return typed dicts or Pydantic models. Log `event=<name>` at entry and exit. If the endpoint needs a component, get it from the registries imported from `bootstrap`.

---

## Design principles and constraints

### Runtime never imports modules
The `coretex/` package must never import from `modules/`. This is the core architectural rule. All coupling goes through the registry lookup pattern.

### Agents never execute tools directly
`ToolExecutor` is the single point of tool execution. Never call a tool function directly from a worker or agent.

### Two LLM calls per request maximum
Classifier (call 1) + worker (call 2). Never add LLM calls to the router, executor, or middleware.

### Router is pure Python
`RouterSimple.route()` must never call an LLM, perform I/O, or make probabilistic decisions. It is a pure dict lookup.

### Stateless requests
Each request starts fresh. No session state, no conversation history, no shared mutable state between requests.

### Worker prompt concatenation
User input is always **appended** via string `+`. Never use `str.format()` or f-strings with user input — user-supplied `{braces}` cause `KeyError`.

### Settings via `coretex/config/settings.py`
All configurable values live in `Settings(BaseSettings)`. Never hardcode URLs, model names, timeouts, or token limits inline.

### Graceful failure over 5xx
The HTTP layer should never return 5xx. Catch `httpx.HTTPError` and tool exceptions at the distribution layer and return HTTP 200 with the appropriate failure response.

### Structured logging
Every event emits `event=<name>` at INFO level. Thread `request_id` through all calls. See `docs/runtime.md` for the full event catalogue.

---

## Running the development server

```bash
pip install -r requirements.txt
uvicorn distributions.cortx.main:app --reload --host 0.0.0.0 --port 8000
```

Enable debug router logging:
```bash
DEBUG_ROUTER=true LOG_LEVEL=DEBUG uvicorn distributions.cortx.main:app --reload
```

---

## Running tests

```bash
# All 106 tests
pytest tests/test_smoke.py -v

# Single test
pytest tests/test_smoke.py::test_executor_tool_action_executes_tool -v

# With coverage
pytest tests/test_smoke.py --cov=coretex --cov=modules --cov-report=term-missing
```

All tests mock Ollama. No running services required.

---

## Further reading

- [`docs/runtime.md`](docs/runtime.md) — runtime internals, pipeline failure catalogue, log event reference
- [`docs/module_development.md`](docs/module_development.md) — complete module authoring guide
- [`docs/distributions.md`](docs/distributions.md) — distribution system, bootstrap pattern, Docker deployment
- [`IMPLEMENTATION.md`](IMPLEMENTATION.md) — implementation reference for AI-assisted development

---

## What comes next (planned phases)

- **Task Graph / Planner** — multi-step orchestration where the planner decomposes requests into a DAG of sub-tasks.
- **Agent collaboration** — multiple agents working on sub-tasks in parallel.
- **Memory layer** — optional context injection for session continuity.
- **Additional tools** — web fetch, shell execution, vector search, database queries.
- **Streaming responses** — progressive output for long-running tasks.
