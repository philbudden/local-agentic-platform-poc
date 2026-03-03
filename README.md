# local-agentic-platform-poc
This is a PoC - not fit for production use

## Getting Started

### Prerequisites
- Docker or Podman with Compose (to run the full stack — **must be run on the host machine, not inside the devcontainer**)
- Python 3.9+ (for tests and running the ingress service directly)
- (Optional) A devcontainer-capable editor (DevPod, VS Code)

### Run the full stack (host machine only)

> ⚠️ The devcontainer runs in DevPod "dockerless" mode — it has no container engine and cannot run `docker compose` or `podman-compose`. Run these commands on your **host machine**.

```bash
# Docker
docker compose up --build

# Podman
podman-compose up --build
```

Then pull a model (first time only):
```bash
docker exec -it ollama ollama pull llama3
# or
podman exec -it ollama ollama pull llama3
```

| Service    | URL                        | Notes                          |
|------------|----------------------------|--------------------------------|
| OpenWebUI  | http://localhost:3000      | Chat interface                 |
| Ingress API| http://localhost:8000      | Internal orchestration API     |
| Ollama     | http://localhost:11434     | Local LLM runtime              |

### Run the ingress service only (devcontainer / no Docker)

To develop and iterate on the ingress API without the full stack:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Note: without Ollama running, `/ingest` calls will hit the network-error path and return `intent=ambiguous`. Use the mock-based tests to work without Ollama.

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

## Architecture

```
User (browser)
  └─► OpenWebUI  (port 3000)
        └─► POST /v1/chat/completions  (Ingress API, port 8000)
              └─► POST /ingest  (internal orchestration)
                    ├─► Classifier (Ollama) → intent + confidence
                    ├─► Router (pure Python) → handler
                    └─► Worker (Ollama) → response
```

See [PLAN.md](PLAN.md) for the full architecture description.
