# PLAN_PHASE1_TAILORED.md

## Local Agentic Platform -- Phase 1 (Tailored to Current Repo)

Date: 2026-03-03

------------------------------------------------------------------------

# Current Repository Structure (Observed)

Root: - Dockerfile - docker-compose.yml - requirements.txt - PLAN.md -
README.md - .devcontainer.json

/app: - main.py - classifier.py - router.py - worker.py - models.py -
settings.py

/tests: - test_smoke.py

Phase 0 confirms: - OpenWebUI → Ingress proxy working - `/ingest`
endpoint stubbed - Docker stack healthy - Smoke tests runnable

------------------------------------------------------------------------

# Phase 1 Objective

Replace stub logic in `app/main.py` with:

Classifier (LLM via Ollama) → Deterministic Router (Python) → Worker
(LLM via Ollama) → Structured response

Maintain strict control-plane separation.

No memory. No tools. No cloud APIs. No architectural expansion beyond
current files.

------------------------------------------------------------------------

# File-by-File Implementation Plan

## 1. app/settings.py

Add explicit configuration:

-   OLLAMA_BASE_URL (default: http://ollama:11434)
-   CLASSIFIER_MODEL (e.g. llama3)
-   WORKER_MODEL (same for now)
-   REQUEST_TIMEOUT
-   MAX_TOKENS

All model names must be configurable via environment variables in
docker-compose.yml.

------------------------------------------------------------------------

## 2. app/models.py

Define strict Pydantic schemas:

ClassifierResponse: - intent:
Literal\["execution","decomposition","novel_reasoning","ambiguous"\] -
confidence: float

IngestRequest: - input: str

IngestResponse: - intent: str - confidence: float - response: str

Validation must fail loudly if classifier JSON invalid.

------------------------------------------------------------------------

## 3. app/classifier.py

Responsibilities: - Call Ollama `/api/generate` - Enforce JSON-only
output - Validate against ClassifierResponse schema - Retry once if JSON
invalid - On second failure → return: intent="ambiguous" confidence=0.0

Classifier must: - Be stateless - Use deterministic temperature
(e.g. 0) - Never include reasoning text

------------------------------------------------------------------------

## 4. app/router.py

Router must be PURE PYTHON.

No LLM calls allowed.

Implement function:

route(intent: str) -\> str

Mapping (Phase 1): - execution → "worker" - decomposition → "worker" -
novel_reasoning → "worker" - ambiguous → "clarify"

Router does not inspect user text.

------------------------------------------------------------------------

## 5. app/worker.py

Responsibilities: - Accept (input_text, intent) - Call Ollama - Return
natural language response

Worker must: - Not modify routing - Not perform classification - Not
persist state - Not call other agents

------------------------------------------------------------------------

## 6. app/main.py

Update `/ingest`:

Flow:

1.  Parse IngestRequest
2.  Call classifier
3.  Call router
4.  If route == "clarify": return clarification template Else: call
    worker
5.  Return IngestResponse

Remove stub logic entirely.

------------------------------------------------------------------------

# docker-compose.yml Updates

Ensure:

-   ollama service exists
-   Ingress service depends_on ollama
-   Environment variables passed to container
-   Ports unchanged

Optional: Add healthcheck for Ollama.

------------------------------------------------------------------------

# Testing Expansion

Extend `tests/test_smoke.py`:

Add:

1.  Test classifier schema validation
2.  Test router deterministic mapping
3.  Test worker returns non-empty string (mock Ollama if needed)
4.  Test full `/ingest` happy path

Keep tests lightweight. No external services required for unit tests.

------------------------------------------------------------------------

# Logging Requirements

Add structured logging in main.py:

Log: - intent - confidence - model latency - total request latency

No tracing stack required.

------------------------------------------------------------------------

# Success Criteria

-   `/ingest` no longer returns stub
-   Real Ollama call confirmed
-   Invalid classifier output handled safely
-   Router contains zero LLM logic
-   All tests pass
-   Total new code \< \~300 lines

------------------------------------------------------------------------

# Guardrails

-   No new directories
-   No planner agent yet
-   No memory tier
-   No tool execution
-   No async complexity

Architecture must remain understandable in \<10 minutes of reading.

------------------------------------------------------------------------

# Phase 2 (Preview Only)

-   Planner agent
-   Budget guardrails
-   Memory tiers
-   Tool-enabled worker
-   Model escalation policies

------------------------------------------------------------------------

Guiding Principle:

LLMs propose. Code disposes.

