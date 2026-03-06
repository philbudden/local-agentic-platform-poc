# Implementation Description — Local Agentic Platform PoC

> This document describes the current state of the implementation as of Phase 3 completion.
> It is intended to be read as context by an LLM when planning future phases of the project.

---

## Purpose

This is a proof-of-concept for a **local agentic platform** that routes user requests through a
two-stage LLM pipeline (classifier → worker) using locally-hosted models via
[Ollama](https://ollama.com). It is not production-ready.

---

## High-Level Architecture

```
User (browser)
  └─► OpenWebUI  (port 3000)
        └─► POST /v1/chat/completions  ← OpenAI-compatible shim
              └─► POST /ingest         ← internal orchestration entry point
                    ├─► Classifier LLM call (Ollama /api/chat)  → intent + confidence
                    ├─► Router (pure Python, no LLM)            → handler name
                    └─► Worker LLM call (Ollama /api/generate)  → free-form response text
```

- **Ollama** runs on the host machine (not in Docker). The ingress container reaches it via
  `host.docker.internal:11434` on Mac/Windows, or via `host-gateway` on Linux.
- **OpenWebUI** is a pre-built chat UI that talks to the ingress API via the OpenAI-compatible
  `/v1/chat/completions` shim.
- **Ingress API** is a FastAPI service that owns all orchestration logic.

---

## Invariants (Architectural Constraints)

These constraints were established in Phase 2 and must remain true:

1. Single orchestration entry point: `POST /ingest`
2. Exactly **two LLM calls** per successful (non-ambiguous) request:
   - Call 1: Classifier → intent classification
   - Call 2: Worker → response generation
3. Router is **pure Python** — no LLM calls, no probabilistic decisions
4. **No memory layer** — each request is stateless
5. **No tool execution** — no external tool calls from within the platform
6. **No autonomous planning loops**

---

## File Structure

```
app/
  __init__.py       — empty package marker
  main.py           — FastAPI app, /ingest endpoint, /v1/chat/completions shim, /debug/routes, /health
  models.py         — Pydantic schemas: ClassifierResponse, IngestRequest, IngestResponse
  classifier.py     — Classifier agent: deterministic prefix checks + LLM call to Ollama
  router.py         — Deterministic intent→handler mapping (pure Python dict lookup)
  worker.py         — Worker agent: intent-aware prompt selection + LLM call to Ollama
  settings.py       — Pydantic-settings config loaded from env vars or .env file
tests/
  test_smoke.py     — All tests (unit + integration via TestClient, Ollama fully mocked)
Dockerfile          — python:3.11-slim, runs uvicorn on port 8000
docker-compose.yml  — ingress + openwebui services on an isolated bridge network
requirements.txt    — fastapi, uvicorn, httpx, pydantic, pydantic-settings, pytest
pytest.ini          — sets pythonpath = . so app.* imports resolve without install
.devcontainer.json  — Python 3 devcontainer, pip installs requirements on create
```

---

## Module Detail

### `app/settings.py`

Pydantic `BaseSettings` subclass. All values are overridable via environment variables
(case-insensitive) or a `.env` file.

| Setting               | Default                              | Purpose                              |
|-----------------------|--------------------------------------|--------------------------------------|
| `ollama_base_url`     | `http://host.docker.internal:11434`  | Base URL for all Ollama API calls    |
| `classifier_model`    | `llama3.2:3b`                        | Model used by the classifier agent   |
| `worker_model`        | `llama3.2:3b`                        | Model used by the worker agent       |
| `classifier_timeout`  | `60` (seconds)                       | httpx timeout for classifier call    |
| `worker_timeout`      | `300` (seconds)                      | httpx timeout for worker call        |
| `max_tokens`          | `256`                                | `num_predict` passed to worker       |
| `ingress_port`        | `8000`                               | Informational only (not used in code)|
| `log_level`           | `INFO`                               | Python logging level                 |
| `debug_router`        | `false`                              | When `true`, log classifier/worker prompts at DEBUG level |

A singleton `settings` instance is imported by all modules.

---

### `app/models.py`

Three Pydantic v2 models:

- **`ClassifierResponse`** — `intent: Literal["execution", "planning", "analysis", "ambiguous"]`,
  `confidence: float`. The `Literal` constraint enforces the valid intent vocabulary.
- **`IngestRequest`** — `input: str`. A `field_validator` rejects empty or whitespace-only strings
  (returns HTTP 422 if violated).
- **`IngestResponse`** — `intent: str`, `confidence: float`, `response: str`. The `intent` field
  here is a plain `str` (not `Literal`) so failure paths can write `"ambiguous"` without schema
  gymnastics.

---

### `app/classifier.py`

The classifier agent is responsible for assigning an intent label and confidence score to
raw user input. It operates in two stages:

**Stage 1 — Deterministic prefix check (no LLM)**

Before calling the LLM, the input is lowercased and matched against hard-coded prefix tuples:

- `_EXECUTION_PREFIXES`: `("write", "generate", "create", "compose", "draft", "produce",
  "summarise", "summarize", "translate", "calculate", "code", "list")` → returns
  `intent="execution", confidence=0.95` immediately.
- `_PLANNING_PREFIXES`: `("how do i", "how would i", "how can i", "what steps")` → returns
  `intent="planning", confidence=0.95` immediately.
- `_AMBIGUOUS_SHORT`: `("help", "hi", "hello", "hey", "ok", "okay", "thanks")` → returns
  `intent="ambiguous", confidence=0.95` immediately.

**Stage 2 — LLM call via `_call_ollama()`**

If no prefix matches, the LLM is called via Ollama `/api/chat` with:
- System prompt (`_SYSTEM_PROMPT`) defining the four intent categories with definitions and
  few-shot examples. User input is passed as a separate user-role message (never interpolated
  into the system prompt — guards against prompt injection).
- Parameters: `temperature=0`, `top_p=0.8`, `num_predict=32`, `format="json"`, `stream=False`.

Up to **2 attempts** are made. On each attempt:
1. Call `_call_ollama()` — raises `httpx.HTTPError` on network/HTTP failure.
2. Call `_parse()` to validate and normalise the response.
3. If `_parse()` returns a valid `ClassifierResponse`, return it immediately.
4. Otherwise log a warning and retry.

After 2 failures, return `ClassifierResponse(intent="ambiguous", confidence=0.0)` as a safe
fallback.

**`_parse()` normalisation logic:**

1. Strip markdown code fences (some models wrap JSON in ` ```json ... ``` `).
2. `json.loads()` the text.
3. Try strict `ClassifierResponse(**data)` — succeeds for well-formed responses.
4. If that fails, scan for the intent value under any of: `intent`, `category`, `type`,
   `classification`, `class`.
5. Normalise the found value: lowercase, replace spaces/hyphens with underscores.
6. Apply `_INTENT_ALIASES` mapping (e.g. `"creative_writing"` → `"execution"`,
   `"novel_reasoning"` → `"analysis"`, `"decomposition"` → `"planning"`).
7. Extract confidence from `confidence`, `score`, or `certainty` fields; default to `0.5`.
8. Attempt `ClassifierResponse(intent=intent, confidence=confidence)` — if validation still
   fails (unrecognised intent), return `ClassifierResponse(intent="ambiguous", confidence=0.0)`.

---

### `app/router.py`

A single dict lookup. No state, no LLM.

```python
ROUTES = {
    "execution": "worker",
    "planning":  "worker",
    "analysis":  "worker",
    "ambiguous": "clarify",
}
```

Any intent not in the dict also maps to `"clarify"`. The handler name (`"worker"` or
`"clarify"`) is returned as a plain string.

---

### `app/worker.py`

The worker agent generates the final user-facing response. It is intent-aware via three prompt
templates stored in `_PROMPTS`:

| Intent      | Prompt directive                                           |
|-------------|-----------------------------------------------------------|
| `execution` | Direct, concise answer; no planning structure; ≤150 words |
| `planning`  | Exactly 3–5 numbered steps; one sentence per step         |
| `analysis`  | Focused, insightful; 3 sentences max                      |

Unknown intents fall back to `_FALLBACK_PROMPT` which is the `execution` template.

The user input is **appended** to the prompt string via concatenation (not `str.format()`),
ensuring that curly braces in user input do not cause `KeyError`.

The LLM call uses Ollama `/api/generate` with `stream=False` and `num_predict=settings.max_tokens`.
The full response text is returned as-is.

`generate()` raises `httpx.HTTPError` on failure; the caller (`main.py`) handles this gracefully.

---

### `app/main.py`

The FastAPI application with three endpoints:

#### `POST /ingest`

The primary orchestration endpoint. Request: `IngestRequest`. Response: `IngestResponse`.

Pipeline:

1. Generate `request_id = uuid4().hex`; log `event=request_received`
2. `classify(request.input, request_id)` → `ClassifierResponse`
3. `route(classifier_result.intent, request_id, user_input, confidence)` → handler name
4. If handler is `"clarify"`: set `response_text = _CLARIFY_RESPONSE` (no LLM call)
5. If handler is `"worker"`:
   - Call `generate(request.input, classifier_result.intent, request_id)`
   - On `httpx.HTTPError`: log `event=worker_error`, set `response_text = _WORKER_FAILURE_RESPONSE`,
     overwrite classifier result to `intent="ambiguous", confidence=0.0`
6. Log `event=request_complete` with `request_id`, `intent`, `confidence`,
   `classifier_latency_ms`, `worker_latency_ms`, `total_latency_ms`
7. Return `IngestResponse`

**Static response strings:**
- `_CLARIFY_RESPONSE`: `"I'm not sure what you're asking. Could you provide more detail or clarify your request?"`
- `_WORKER_FAILURE_RESPONSE`: `"I'm sorry, I was unable to process your request right now. Please try again later."`

#### `POST /v1/chat/completions`

OpenAI-compatible shim for OpenWebUI. Accepts a standard ChatCompletion request body
(model, messages list, stream flag). Extracts the last `role="user"` message and forwards
it to the `/ingest` logic. Wraps the result in a minimal ChatCompletion-shaped response with:
- `object: "chat.completion"`
- `choices[0].message.role = "assistant"`
- `choices[0].message.content = result.response`
- `usage` tokens all set to 0 (not tracked)
- A random `chatcmpl-{uuid4}` ID

If no user message is present or it is whitespace-only, returns the clarification response
without calling ingest.

#### `GET /debug/routes`

Returns `{"routes": ROUTES}` — the live intent→handler mapping dict. Intended for development
inspection and validating new intent additions.

#### `GET /health`

Returns `{"status": "ok"}`. Used for liveness checks.

---

## Testing

All tests are in `tests/test_smoke.py`. They run against FastAPI's `TestClient` — no Docker
or running Ollama required. Ollama calls are mocked via `unittest.mock.patch` and `AsyncMock`.

The test suite covers (as of Phase 3):

| Category                          | Tests                                                              |
|-----------------------------------|--------------------------------------------------------------------|
| Router unit tests                 | All 4 intent→handler mappings + unknown intent fallback           |
| Classifier schema validation      | Valid schema accepted; invalid intent rejected by Pydantic        |
| Classifier behaviour (async unit) | Network error fallback, invalid JSON fallback, markdown fence     |
|                                   | stripping, alias normalisation, alternative field names,          |
|                                   | capitalisation normalisation                                       |
| `/ingest` happy path              | Success, ambiguous, missing/empty/whitespace input (422),         |
|                                   | curly braces in input                                             |
| `/v1/chat/completions` shim       | 200 response, empty messages, whitespace-only message            |
| Health check                      | `GET /health` → 200                                               |
| Worker prompts (Phase 2)          | execution/planning/analysis prompts contain expected keywords,    |
|                                   | unknown intent uses fallback                                       |
| Worker failure handling           | `ConnectError` and `TimeoutException` both return 200 with        |
|                                   | `intent="ambiguous", confidence=0.0`                              |
| Phase 3: observability            | `GET /debug/routes` returns ROUTES dict; router logs              |
|                                   | `event=router_fallback` for unknown intents and                   |
|                                   | `event=intent_router` for every decision; classifier logs         |
|                                   | `event=classifier_result` for prefix-match hits                   |

**Test runner:** `pytest tests/test_smoke.py -v`

Async tests use `@pytest.mark.anyio` (requires `anyio` to be available; it is installed
transitively via `httpx`).

---

## Infrastructure

### Dockerfile

- Base image: `python:3.11-slim`
- Workdir: `/workspace`
- Copies `requirements.txt` then `app/` only (tests are not included in the image)
- Entry point: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### `docker-compose.yml`

Two services on an isolated `agentic` bridge network:

| Service    | Image                                  | Port mapping   | Notes                              |
|------------|----------------------------------------|----------------|------------------------------------|
| `ingress`  | Built from `Dockerfile`                | `8000:8000`    | All env vars configurable          |
| `openwebui`| `ghcr.io/open-webui/open-webui:main`   | `3000:8080`    | `OPENAI_API_BASE_URL=http://ingress:8000/v1` |

The `ingress` service uses `extra_hosts: host.docker.internal:host-gateway` to allow
the container to reach Ollama on the host on Linux (Docker Desktop handles this automatically
on Mac/Windows).

OpenWebUI has `ENABLE_OLLAMA_API=false` to ensure it only uses the ingress shim.

### Devcontainer

- Image: `mcr.microsoft.com/devcontainers/python:3`
- `postCreateCommand`: `pip install -r requirements.txt`
- Forwards port 8000
- Intended for editing and running tests; `docker compose` cannot be run inside the devcontainer
  (no container engine available in dockerless DevPod mode)

---

## What Has Been Done (Phase 1 + Phase 2 + Phase 3)

- **Phase 1**: Skeleton FastAPI service, `/ingest` endpoint, classifier LLM call, deterministic
  router, basic worker LLM call, smoke tests, Docker/Compose setup, OpenWebUI integration.
- **Phase 2**: Intent-aware worker prompts (execution/planning/analysis templates), classifier
  hardening (deterministic prefix checks, alias normalisation, alternative field handling,
  markdown fence stripping, retry-with-fallback), observability logging (per-request timing),
  graceful worker failure handling (HTTP errors and timeouts return 200 with failure envelope),
  expanded test suite covering all new behaviour.
- **Phase 3**: Request correlation IDs (`request_id=uuid4().hex` generated per `/ingest` call,
  passed through classify/route/generate), structured log events (`event=` prefix on all log
  messages: `request_received`, `classifier_result`, `classifier_retry`, `classifier_latency`,
  `classifier_raw_output`, `classifier_fallback`, `intent_router`, `router_fallback`,
  `worker_start`, `worker_complete`, `request_complete`, `worker_error`), `DEBUG_ROUTER` env
  var (logs classifier and worker prompts at DEBUG when `true`), `GET /debug/routes` endpoint
  returning the live routing table, latency units changed from seconds to milliseconds.

---

## What Has NOT Been Built (Non-Goals so far)

- Memory / conversation history
- Streaming responses
- Tool use / function calling
- Multi-agent coordination
- Authentication / authorisation
- Persistent storage of any kind
- Rate limiting
- Multiple model backends (only Ollama supported)
- Async concurrency beyond what FastAPI/httpx provide naturally
- Separate classifier and worker model instances (both default to the same model)
