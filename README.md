# :brain: Cortex

> **This project is in Alpha - it may introduce breaking changes** | **This project is not production ready**

**Cortex** is a local-first intelligent automation platform designed to turn natural language into reliable, structured system behaviour.

Rather than being a single model, service, or workflow engine, Cortex is an orchestration layer that connects language understanding, structured reasoning, and tool execution into a cohesive system. Its purpose is simple: allow humans to describe *what they want*, while Cortex determines *how to accomplish it* safely and deterministically.

At its core, Cortex is built around a clear principle: **language should be an interface, not the system itself**.

Large language models are powerful interpreters of intent, but they are not inherently reliable decision engines. Cortex separates interpretation from execution, using structured routing, deterministic validation, and controlled tool interfaces to convert ambiguous human input into predictable outcomes.

The platform is designed to run **locally and privately**, allowing individuals, engineers, and organisations to build intelligent systems without depending on external APIs or opaque infrastructure. Every component — from the language models to the orchestration layer — is intended to be deployable within environments you control.

Over time, Cortex aims to evolve into a foundation for building intelligent software systems where:

- Natural language becomes a **first-class interface**
- Automation remains **transparent and debuggable**
- AI behaviour is **observable and auditable**
- Tools and services can be **safely composed and extended**
- Systems remain **local-first, modular, and developer-friendly**

The long-term vision of Cortex is to provide a platform where intelligent behaviour emerges from **clear architecture rather than prompt engineering** — enabling developers to build systems that are understandable, reliable, and adaptable as language models continue to evolve.

In this model, Cortex is not the intelligence itself.

It is the **cognitive layer** that allows intelligence, software, and real-world actions to work together.

---

## Architecture

```
User (browser)
  └─► OpenWebUI  (port 3000)
        └─► POST /v1/chat/completions  (Ingress API, port 8000)
              └─► POST /ingest  (internal orchestration)
                    ├─► Classifier  — LLM call 1/2 → intent + confidence
                    ├─► Router      — pure Python dict lookup → handler
                    └─► Worker      — LLM call 2/2 → response text
```

- **Classifier** — calls Ollama, returns one of `execution | planning | analysis | ambiguous`. Deterministic prefix checks short-circuit common patterns before any LLM call.
- **Router** — a Python dict. Given the same intent, always returns the same handler. No LLM involved.
- **Worker** — selects an intent-aware prompt template, calls Ollama, returns free-form text.
- **OpenWebUI** — UI only. `ENABLE_OLLAMA_API=false`. It cannot bypass the pipeline.

Ollama runs on the host machine, not in Docker. The container reaches it via `host.docker.internal:11434`.

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
| `DEBUG_ROUTER`     | `false`                             | When `true`, logs classifier and worker prompts at DEBUG     |

`docker-compose.yml` uses `${VAR:-default}` interpolation throughout — shell variables always take precedence over defaults without editing the file.

---

## Observability

Every request gets a `request_id`. All log lines carry `event=<name>` and `request_id=<id>`.

```bash
# Follow live
docker compose logs -f ingress

# Trace a single request
docker compose logs ingress | grep "request_id=<id>"
```

**Typical log sequence:**
```
event=request_received      request_id=<id>
event=llm_call              request_id=<id> call=1/2 model=llama3.2:3b
event=classifier_result     request_id=<id> intent=analysis confidence=0.90 source=llm
event=classifier_latency    request_id=<id> latency_ms=1820
event=intent_router         request_id=<id> intent=analysis route=worker confidence=0.90
event=worker_start          request_id=<id> worker=worker intent=analysis
event=llm_call              request_id=<id> call=2/2 model=llama3.2:3b intent=analysis
event=worker_complete       request_id=<id> worker=worker latency_ms=3240
event=request_complete      request_id=<id> intent=analysis confidence=0.90 total_latency_ms=5063
```

For inputs matching a prefix (`Write...`, `How do I...`) the classifier short-circuits — no `llm_call 1/2` or `classifier_latency` appears, and `source=prefix_match`.

**Enable prompt logging:**
```bash
DEBUG_ROUTER=true LOG_LEVEL=DEBUG docker compose up --build
```

**Inspect the routing table:**
```bash
curl http://localhost:8000/debug/routes
# → {"routes":{"execution":"worker","planning":"worker","analysis":"worker","ambiguous":"clarify"}}
```

