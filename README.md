# :brain: COREtex

> **This project is in Alpha - it may introduce breaking changes** | **This project is not production ready**

**CortX AI** is a local-first intelligent automation platform designed to turn natural language into reliable, structured system behaviour.

Rather than being a single model, service, or workflow engine, CortX is an orchestration layer that connects language understanding, structured reasoning, and tool execution into a cohesive system. Its purpose is simple: allow humans to describe *what they want*, while CortX AI determines *how to accomplish it* safely and deterministically.

At its core, CortX AI is built around a clear principle: **language should be an interface, not the system itself**.

Large language models are powerful interpreters of intent, but they are not inherently reliable decision engines. CortX AI separates interpretation from execution, using structured routing, deterministic validation, and controlled tool interfaces to convert ambiguous human input into predictable outcomes.

The platform is designed to run **locally and privately**, allowing individuals, engineers, and organisations to build intelligent systems without depending on external APIs or opaque infrastructure. Every component — from the language models to the orchestration layer — is intended to be deployable within environments you control.

Over time, CortX AI aims to evolve into a foundation for building intelligent software systems where:

- Natural language becomes a **first-class interface**
- Automation remains **transparent and debuggable**
- AI behaviour is **observable and auditable**
- Tools and services can be **safely composed and extended**
- Systems remain **local-first, modular, and developer-friendly**

---

## What CortX AI can do today (v0.4.x)

- **Understand intent** — classifies every request as `execution`, `planning`, `analysis`, or `ambiguous` using a local LLM, with deterministic prefix checks for common patterns.
- **Route deterministically** — maps intent to the correct execution path using a pure Python dict, not another LLM.
- **Generate structured responses** — selects an intent-aware prompt template, calls the worker LLM, and returns a response tailored to the type of request.
- **Execute tools** — agents return structured JSON specifying either a direct reply or a tool to run; the ToolExecutor carries out the action safely.
- **Read files** — the built-in `read_file` tool reads any local file by path and returns its contents.
- **Observe everything** — every request gets a unique ID; every step emits a structured `event=<name> key=value` log including `event=pipeline_selected`.
- **Run configurable pipelines** — the `PipelineRegistry` holds named `PipelineDefinition` objects; the `PipelineRunner` dynamically resolves components from the definition at runtime.
- **Load components as modules** — classifiers, routers, workers, and tools are registered dynamically at startup via the module loader, with signature validation and lifecycle events.
- **Fail gracefully** — classifier failures, worker failures, tool lookup errors, and tool exceptions all produce safe fallback responses, never unhandled 500s.

---

## Architecture (v0.4.x)

COREtex v0.4 is structured as a **runtime platform** with three layers:

```
coretex/              ← Runtime platform
  runtime/          ← PipelineRunner, PipelineDefinition, PipelineStep, ToolExecutor, ModuleLoader, ExecutionContext, EventBus
  interfaces/       ← ABCs: Classifier, Router, Worker, ModelProvider
  registry/         ← ToolRegistry, ModuleRegistry, ModelProviderRegistry, PipelineRegistry
  config/           ← Settings

modules/            ← Components implementing interfaces, registered at startup
  classifier_basic/ ← Intent classifier (prefix checks + LLM)
  router_simple/    ← Deterministic dict-based router
  worker_llm/       ← LLM response generator
  tools_filesystem/ ← read_file tool
  model_provider_ollama/ ← Ollama inference backend

distributions/
  cortx/      ← FastAPI ingress + OpenWebUI integration

docs/               ← Runtime, module development, and distributions guides
```

### Pipeline system (v0.4.0)

Pipelines are now first-class objects. The `PipelineRegistry` holds named `PipelineDefinition` instances, each describing an ordered sequence of `PipelineStep` objects. The `PipelineRunner` reads the definition at startup to determine which named components to use.

```python
from coretex.runtime.pipeline import PipelineDefinition, PipelineStep

my_pipeline = PipelineDefinition(
    name="my_pipeline",
    steps=[
        PipelineStep(component_type="classifier", name="classifier_basic"),
        PipelineStep(component_type="router",     name="router_simple"),
        PipelineStep(component_type="worker",     name="worker_llm"),
        PipelineStep(component_type="tool_executor", name="tool_executor"),
    ],
)
```

The default pipeline (`"default"`) preserves the pre-v0.4.0 behaviour exactly.

### Request pipeline

```
User (browser)
  └─► OpenWebUI  (port 3000)
        └─► POST /v1/chat/completions  (cortx, port 8000)
              └─► POST /ingest  (internal orchestration via PipelineRunner)
                    │  pipeline_selected log (pipeline=default)
                    ├─► Classifier  — LLM call 1/2 → ClassificationResult
                    ├─► Router      — pure Python dict lookup → handler
                    └─► Worker      — LLM call 2/2 → JSON action envelope
                                         │
                                    Action Parser
                                         │
                                    Tool Executor  → Tool Result
```

- **Classifier** — calls Ollama, returns one of `execution | planning | analysis | ambiguous`. Deterministic prefix checks short-circuit common patterns before any LLM call.
- **Router** — a Python dict. Given the same intent, always returns the same handler. No LLM involved.
- **Worker** — selects an intent-aware prompt template, calls Ollama, and returns a JSON action envelope.
- **Action Parser** — parses the agent's JSON output into a typed `AgentAction`.
- **Tool Executor** — the only component that can run tools. Looks up the tool by name in the `ToolRegistry` and calls it deterministically. Agents never execute tools directly.
- **OpenWebUI** — UI only. `ENABLE_OLLAMA_API=false`. It cannot bypass the pipeline.

Ollama runs on the host machine, not in Docker. The container reaches it via `host.docker.internal:11434`.

### Agent output contract

Agents (the worker LLM) must return strict JSON. Two formats are supported:

**Direct reply:**
```json
{"action": "respond", "content": "Here is the answer."}
```

**Tool call:**
```json
{"action": "tool", "tool": "read_file", "args": {"path": "notes.md"}}
```

If the LLM returns plain text instead of JSON, COREtex gracefully falls back to treating it as a direct response.

---

## Quick start

**Prerequisites:** [Ollama](https://ollama.com) running on the host, Docker or Podman with Compose.

```bash
# 1. Pull a model
ollama pull llama3.2:3b

# 2. Start the stack
docker compose up --build
```

| Service     | URL                    |
|-------------|------------------------|
| OpenWebUI   | http://localhost:3000  |
| Ingress API | http://localhost:8000  |

```bash
# 3. Send a request
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"input": "Compare Kubernetes and Nomad"}'
# → {"intent":"analysis","confidence":0.9,"response":"..."}

# 4. Request file reading via tool call
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"input": "Read the file /etc/hostname"}'
```

> ⚠️ If your input contains an apostrophe (`I'm`, `don't`), it will close the shell string and curl will appear to freeze. Use `'\''` to escape or write the payload to a file: `-d @body.json`

**Use a remote Ollama instance:**
```bash
OLLAMA_BASE_URL=http://192.168.1.50:11434 docker compose up --build
```

**Change models:**
```bash
CLASSIFIER_MODEL=llama3.2:3b WORKER_MODEL=llama3.1:8b docker compose up --build
```

**OpenWebUI:** Browse to http://localhost:3000, create a local account, select the **agentic** model from the dropdown, and type any message.

> **Single-turn only:** The `/v1/chat/completions` shim extracts only the most recent user message. Prior turns are visible in the OpenWebUI chat history but are not sent to the API — each request is processed independently. This is deliberate.

**Run tests (no Docker required):**
```bash
pip install -r requirements.txt
pytest tests/test_smoke.py -v
```

---

## Configuration

All settings are overridable via environment variables or a `.env` file.

| Variable           | Default                             | Purpose                                                      |
|--------------------|-------------------------------------|--------------------------------------------------------------|
| `OLLAMA_BASE_URL`  | `http://host.docker.internal:11434` | Ollama endpoint                                              |
| `CLASSIFIER_MODEL` | `llama3.2:3b`                       | Model used for intent classification                         |
| `WORKER_MODEL`     | `llama3.2:3b`                       | Model used for response generation                           |
| `CLASSIFIER_TIMEOUT` | `60`                              | Seconds before classifier call times out                     |
| `WORKER_TIMEOUT`   | `300`                               | Seconds before worker call times out                         |
| `MAX_TOKENS`       | `256`                               | Max tokens generated by the worker                           |
| `LOG_LEVEL`        | `INFO`                              | `DEBUG`, `INFO`, or `WARNING`                                |
| `DEBUG_ROUTER`     | `false`                             | When `true`, logs `event=router_decision` at DEBUG level     |

`docker-compose.yml` uses `${VAR:-default}` interpolation throughout — shell variables always take precedence over defaults without editing the file.

---

## Observability

Every request gets a `request_id`. All log lines carry `event=<name>` and `request_id=<id>` in structured `key=value` format.

```bash
# Follow live
docker compose logs -f ingress

# Trace a single request
docker compose logs ingress | grep "request_id=<id>"
```

**Typical log sequence (with tool execution):**
```
event=pipeline_selected     request_id=<id> pipeline=default
event=request_received      request_id=<id>
event=classifier_start      request_id=<id> classifier=classifier_basic
event=classifier_complete   request_id=<id> intent=execution confidence=0.95 duration_ms=312
event=router_selected       request_id=<id> intent=execution handler=worker
event=worker_start          request_id=<id> worker=worker_llm intent=execution
event=worker_complete       request_id=<id> duration_ms=1450
event=agent_output_received request_id=<id>
event=tool_execute          tool=read_file  request_id=<id>
event=tool_execute_complete tool=read_file  request_id=<id>
event=request_complete      request_id=<id> intent=execution confidence=0.95 handler=worker total_latency_ms=1765
```

**Enable debug router logging:**
```bash
DEBUG_ROUTER=true LOG_LEVEL=DEBUG docker compose up --build
```

**Inspect the routing table:**
```bash
curl http://localhost:8000/debug/routes
# → {"routes":{"execution":"worker","planning":"worker","analysis":"worker","ambiguous":"clarify"}}
```

---

## Further Reading

- [`docs/runtime.md`](docs/runtime.md) — runtime architecture, pipeline, and failure behaviour
- [`docs/module_development.md`](docs/module_development.md) — how to build new modules
- [`docs/distributions.md`](docs/distributions.md) — how to build a distribution
- [`DEVELOPMENT.md`](DEVELOPMENT.md) — developer guide for extending the project
- [`TESTING.md`](TESTING.md) — how to validate the system
- [`IMPLEMENTATION.md`](IMPLEMENTATION.md) — full implementation description
