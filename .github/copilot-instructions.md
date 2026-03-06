# Copilot Instructions

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/test_smoke.py -v

# Run a single test
pytest tests/test_smoke.py::test_ingest_happy_path -v

# Run the ingress service locally (no Docker required)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run the full stack (host machine only — not inside devcontainer)
docker compose up --build

# Enable debug prompt logging
DEBUG_ROUTER=true LOG_LEVEL=DEBUG docker compose up --build

# Follow ingress logs live
docker compose logs -f ingress

# Trace a single request by correlation ID
docker compose logs ingress | grep "request_id=<id>"

# Inspect live routing table
curl http://localhost:8000/debug/routes
```

## Architecture

This is a **two-stage LLM pipeline** (classifier → worker) backed by a locally-hosted Ollama instance. All orchestration lives in the FastAPI ingress service (`app/`):

```
POST /ingest
  ├─ classify(input, request_id)                → ClassifierResponse(intent, confidence)   [LLM call 1]
  ├─ route(intent, request_id, user_input, confidence) → handler ("worker" | "clarify")    [pure Python]
  └─ generate(input, intent, request_id)        → str                                       [LLM call 2, skipped if ambiguous]
```

**Invariants** — these must never be violated:
- Exactly **2 LLM calls** per non-ambiguous request (1 classifier + 1 worker). `ambiguous` intent short-circuits and skips the worker call entirely.
- The router (`app/router.py`) is **pure Python** — no LLM, no probabilistic logic, just a dict lookup.
- Each request is **stateless** — no memory, no conversation history, no tool calls.
- The single orchestration entry point is `POST /ingest`. The `/v1/chat/completions` endpoint is a thin OpenAI-compatible shim that extracts the last user message and delegates to `/ingest`.
- `GET /debug/routes` returns the live `ROUTES` dict for development inspection.

**Ollama** runs on the host machine (not in Docker). The ingress container reaches it via `host.docker.internal:11434`. Override with `OLLAMA_BASE_URL`.

## Key Conventions

### Structured logging with correlation IDs
Every request generates a `request_id = uuid4().hex` in `main.py` which is threaded through `classify()`, `route()`, and `generate()`. All log messages use `event=<name>` as the first field followed by `request_id=<id>` for correlation. Standard log events in sequence: `request_received` → `classifier_result` → `intent_router` → `worker_start` → `worker_complete` → `request_complete`. Set `DEBUG_ROUTER=true` to additionally emit `classifier_prompt` and `worker_prompt` at DEBUG level.

### Classifier: deterministic prefix checks before LLM
`app/classifier.py` runs a cheap string-prefix match before any LLM call. Inputs starting with execution verbs (`write`, `generate`, `create`, …) return `intent="execution", confidence=0.95` immediately without calling Ollama. Add new prefix sets here before considering LLM-side changes.

### Intent vocabulary
The four valid intents are enforced as a `Literal` in `ClassifierResponse`: `"execution"`, `"planning"`, `"analysis"`, `"ambiguous"`. `IngestResponse.intent` is a plain `str` to allow the failure path to write `"ambiguous"` without gymnastics — this asymmetry is intentional.

### Worker prompt templating uses concatenation, not `str.format()`
User input is **appended** to the prompt string via `+`. Never switch to `str.format()` or f-strings here — user-supplied curly braces would cause `KeyError`.

### `_parse()` normalisation pipeline
The classifier's `_parse()` function handles LLM quirks in a specific order: strip markdown fences → `json.loads` → strict Pydantic parse → fallback field-name scan (`_INTENT_FIELD_CANDIDATES`) → alias map (`_INTENT_ALIASES`) → graceful `ambiguous` fallback. Follow this pattern when extending classifier robustness.

### Retry-with-fallback pattern
`classify()` retries the LLM call once on any failure (network error or bad JSON). After 2 failures it returns `ClassifierResponse(intent="ambiguous", confidence=0.0)`. The caller never raises; errors are absorbed into the failure envelope.

### Configuration via Pydantic Settings
All tuneable values live in `app/settings.py` (`Settings(BaseSettings)`). They are overridable by environment variable or `.env` file. A singleton `settings` is imported by all modules — never hardcode URLs, model names, or timeouts inline.

### Tests: mock at the `app.main` boundary
Tests patch `app.main.classify` and `app.main.generate` (not the Ollama HTTP client directly) using `AsyncMock`. Unit tests for `_parse()` and `route()` can call those functions directly without any mocking. Async tests use `@pytest.mark.anyio`.

### Graceful failure contract
`generate()` is allowed to raise `httpx.HTTPError`; `main.py` catches it and returns HTTP 200 with `intent="ambiguous", confidence=0.0` and a polite `_WORKER_FAILURE_RESPONSE` string. Never let LLM errors propagate as 5xx responses.
