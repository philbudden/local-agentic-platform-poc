# local-agentic-platform-poc
This is a PoC - not fit for production use

## Prerequisites

- **Ollama** — must be installed and running on the host machine before starting the stack.
  Download from https://ollama.com and pull a model, e.g.:
  ```bash
  ollama pull llama3.2:3b
  ```
  Ollama must be listening on its default port (`11434`). GPU acceleration is configured by Ollama itself — see the Ollama docs for your platform.
- **Docker or Podman with Compose** — to run the stack (**must be run on the host machine, not inside the devcontainer**)
- **Python 3.9+** — for tests and running the ingress service directly
- (Optional) A devcontainer-capable editor (DevPod, VS Code)

## Getting Started

### Run the full stack (host machine only)

> ⚠️ The devcontainer runs in DevPod "dockerless" mode — it has no container engine and cannot run `docker compose` or `podman-compose`. Run these commands on your **host machine**.

```bash
# Docker
docker compose up --build

# Podman
podman-compose up --build
```

| Service     | URL                    | Notes                      |
|-------------|------------------------|----------------------------|
| OpenWebUI   | http://localhost:3000  | Chat interface             |
| Ingress API | http://localhost:8000  | Internal orchestration API |
| Ollama      | http://localhost:11434 | Runs on host (not in Docker) |

### Use a remote Ollama instance

Set `OLLAMA_BASE_URL` to point at any Ollama instance before starting:

```bash
OLLAMA_BASE_URL=http://192.168.1.50:11434 docker compose up --build
```

Or edit `OLLAMA_BASE_URL` directly in `docker-compose.yml`.

### Configure the model

The classifier and worker can use different models. Override via env vars or edit `docker-compose.yml`:

```bash
CLASSIFIER_MODEL=llama3.2:3b WORKER_MODEL=llama3.2:3b docker compose up --build
```

Both default to `llama3.2:3b`. Any model available in your Ollama installation can be used — pull it first with `ollama pull <model>`.

| Variable           | Default        | Purpose                    |
|--------------------|----------------|----------------------------|
| `OLLAMA_BASE_URL`  | `http://host.docker.internal:11434` | Ollama endpoint |
| `CLASSIFIER_MODEL` | `llama3.2:3b`  | Intent classification      |
| `WORKER_MODEL`     | `llama3.2:3b`  | Response generation        |
| `LOG_LEVEL`        | `INFO`         | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `DEBUG_ROUTER`     | `false`        | When `true`, logs classifier and worker prompts at DEBUG level |

### Run the ingress service only (devcontainer / no Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> Without Ollama running, `/ingest` calls will hit the network-error path and return `intent=ambiguous`. Use the mock-based tests to work without Ollama.

### Verify the endpoint

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"input": "Write a haiku about databases"}'
# → {"intent":"execution","confidence":0.92,"response":"..."}
```

### OpenWebUI

1. Browse to http://localhost:3000 and create a local account.
2. Type any message — it routes through the ingress API to Ollama.

### Run smoke tests (no Docker required)

```bash
pip install -r requirements.txt
pytest tests/test_smoke.py -v
```

---

## Observability

### Checking logs

Each request is assigned a unique `request_id`. All log lines include `event=<name>` and `request_id=<id>` so you can trace a single request end-to-end.

**Docker (follow live):**
```bash
docker compose logs -f ingress
```

**Filter a single request by ID:**
```bash
docker compose logs ingress | grep "request_id=<id>"
```

**Expected log sequence for a successful request:**
```
event=request_received request_id=<id>
event=llm_call request_id=<id> call=1/2 ...
event=classifier_result request_id=<id> intent=execution confidence=0.95 source=llm
event=classifier_latency request_id=<id> latency_ms=...
event=intent_router request_id=<id> intent=execution route=worker ...
event=worker_start request_id=<id> worker=worker intent=execution
event=llm_call request_id=<id> call=2/2 ...
event=worker_complete request_id=<id> worker=worker latency_ms=...
event=request_complete request_id=<id> intent=execution confidence=... total_latency_ms=...
```

### Enable debug logging

Set `DEBUG_ROUTER=true` to log the classifier system prompt and worker prompt at `DEBUG` level:

```bash
DEBUG_ROUTER=true LOG_LEVEL=DEBUG docker compose up --build
```

### Inspect the routing table

```bash
curl http://localhost:8000/debug/routes
# → {"routes":{"execution":"worker","planning":"worker","analysis":"worker","ambiguous":"clarify"}}
```

---

## Architecture

```
User (browser)
  └─► OpenWebUI  (port 3000)
        └─► POST /v1/chat/completions  (Ingress API, port 8000)
              └─► POST /ingest  (internal orchestration)
                    ├─► Classifier (Ollama on host) → intent + confidence
                    ├─► Router (pure Python) → handler
                    └─► Worker (Ollama on host) → response
```

Ollama runs on the host machine. The ingress container reaches it via
`host.docker.internal:11434` (Docker Desktop on Mac/Windows) or via the
`host-gateway` extra_host on Linux. Override with `OLLAMA_BASE_URL` for
remote deployments.

See [PLAN.md](PLAN.md) for the full architecture description.
