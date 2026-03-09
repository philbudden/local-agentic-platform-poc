# Testing Guide

This guide explains how to validate that all functionality of COREtex is working correctly — from the automated test suite through to end-to-end live requests.

---

## Prerequisites

- Python 3.9 or later
- [Ollama](https://ollama.com) (required for live requests; **not** required to run the test suite)
- Docker or Podman with Compose (required for the full stack only)

---

## 1. Run the automated test suite (no Docker, no Ollama required)

The entire test suite mocks all Ollama calls, so it runs in any environment with Python installed.

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/test_smoke.py -v
```

Expected output: **106 tests passed**.

### What the tests cover

| Category | What is verified |
|----------|-----------------|
| Router | All 4 intent→handler mappings; unknown-intent fallback to `clarify`; `router_fallback` log event |
| Classifier schema | Valid and invalid intent values (`_ClassifierResponse` internal validator) |
| Classifier behaviour | Network error fallback, invalid JSON fallback, markdown-fence stripping, alias normalisation, alternative field names, capitalisation normalisation |
| `/ingest` pipeline | Happy path, ambiguous input, missing/empty/whitespace input (422), curly braces in input |
| `/v1/chat/completions` | 200 response, empty messages, whitespace-only messages |
| Health and models | `GET /health` → 200; `GET /v1/models` returns `agentic` |
| Worker prompts | Each intent's prompt contains expected keywords; unknown intent uses fallback; JSON instructions present |
| Worker failure | `ConnectError` and `TimeoutException` both return 200 with `intent="ambiguous"` |
| Observability | `GET /debug/routes`; `router_fallback` and `intent_router` log events; `classifier_result` for prefix matches |
| Tool registry | Register, get, duplicate raises (`already registered`), unknown-tool raises (`Unknown component`), list |
| Tool execution | `Tool.execute()` calls the underlying function |
| AgentAction | `from_dict` for respond and tool actions; args defaults to `{}` |
| ToolExecutor | respond action returns content; tool action executes tool; unknown action raises; unknown tool raises; missing tool name raises; runtime exception propagates |
| `parse_agent_output` | Valid respond JSON; valid tool JSON; invalid JSON raises `JSONDecodeError` |
| Filesystem tool | Reads existing file; returns error string for missing file |
| Module bootstrap | `read_file` is registered at startup |
| **Registry validation** | Duplicate classifier/router/worker/model_provider/pipeline raises; unknown classifier/router/worker/model_provider/pipeline raises; `registry_lookup_failed` log emitted |
| **ModuleLoader** | Valid module loads; missing `register()` raises; wrong signature raises; empty registration logs warning; `ImportError` raises; `load_all()` emits lifecycle events |
| **Pipeline failures** | Classifier HTTP failure returns clarification; worker HTTP failure returns graceful response; invalid JSON treated as plain text; tool lookup failure returns failure response; tool runtime exception returns failure response |
| **Logging events** | `request_received`, `classifier_complete`, `router_selected`, `worker_complete`, `request_complete` all present; `total_latency_ms` and `duration_ms` in appropriate events |
| **ExecutionContext** | `timestamp` is a float; `metadata` defaults to None; `metadata` can be set |
| **Router debug** | `router_decision` logged at DEBUG when `debug_router=True`; not logged when `debug_router=False` |
| Integration | JSON respond envelope unwrapped; tool call reads real temp file; plain-text fallback; unknown tool returns failure; tool execution logs carry `request_id` |

---

## 2. Start the system locally (without Docker)

```bash
pip install -r requirements.txt
uvicorn distributions.cortx.main:app --reload --host 0.0.0.0 --port 8000
```

This starts the ingress API on port 8000. Ollama must be running on the host for real requests to work.

---

## 3. Start the full stack (Docker)

```bash
# Pull a model first
ollama pull llama3.2:3b

# Start everything
docker compose up --build
```

| Service | URL |
|---------|-----|
| Ingress API | http://localhost:8000 |
| OpenWebUI | http://localhost:3000 |

---

## 4. Verify routing behaviour

### Check the routing table

```bash
curl http://localhost:8000/debug/routes
```

Expected response:
```json
{"routes":{"execution":"worker","planning":"worker","analysis":"worker","ambiguous":"clarify"}}
```

### Send an ambiguous request (should clarify)

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"input": "hello"}' | python3 -m json.tool
```

Expected: `intent` = `"ambiguous"`, response contains "clarify" or "detail".

### Send a planning request

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"input": "How do I deploy a Docker container?"}' | python3 -m json.tool
```

Expected: `intent` = `"planning"`, response contains numbered steps.

### Send an analysis request

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"input": "Compare Kubernetes and Nomad"}' | python3 -m json.tool
```

Expected: `intent` = `"analysis"`, response contains a concise comparison.

### Send an execution request

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"input": "Write a haiku about rain"}' | python3 -m json.tool
```

Expected: `intent` = `"execution"`, `source=prefix_match` in logs (no LLM call for classification), response contains a haiku.

---

## 5. Verify tool execution

### Read a local file via the tool

```bash
echo "This is my test file content." > /tmp/coretex_test.txt

curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"input": "Read the file at /tmp/coretex_test.txt and tell me what it says"}' \
  | python3 -m json.tool
```

Expected: the response contains `"This is my test file content."`.

> **Docker:** `docker-compose.yml` mounts the host's `/tmp` directory into the container read-only. Create the file on the host before sending the request.

> **Note:** Whether the LLM produces a tool-call JSON or a direct response depends on model behaviour. A well-prompted model (llama3.2:3b or better) should emit the tool call. If it does not, the response will be the model's best guess without file access.

### Directly test the tool execution layer (Python REPL)

```python
from distributions.cortx.bootstrap import tool_registry
from coretex.runtime.executor import ToolExecutor, AgentAction, parse_agent_output

executor = ToolExecutor(tool_registry)

# Direct respond action
action = AgentAction(action="respond", content="Hello from executor")
print(executor.execute(action))  # → "Hello from executor"

# Tool action — reads a real file
import json, tempfile, pathlib
f = pathlib.Path(tempfile.mktemp(suffix=".txt"))
f.write_text("file content here")
raw = json.dumps({"action": "tool", "tool": "read_file", "args": {"path": str(f)}})
action = parse_agent_output(raw)
print(executor.execute(action))  # → "file content here"
```

### Missing file returns an error string (not a 500)

```python
from distributions.cortx.bootstrap import tool_registry
from coretex.runtime.executor import AgentAction, ToolExecutor

executor = ToolExecutor(tool_registry)
action = AgentAction(action="tool", tool="read_file", args={"path": "/nonexistent/file.txt"})
print(executor.execute(action))  # → "File not found: /nonexistent/file.txt"
```

---

## 6. Verify logging output

### Enable debug router logging

```bash
DEBUG_ROUTER=true LOG_LEVEL=DEBUG docker compose up --build
```

### Follow logs live

```bash
docker compose logs -f ingress
```

### Expected log sequence for a tool call request

```
event=request_received      request_id=<id>
event=classifier_start      request_id=<id> classifier=classifier_basic
event=classifier_complete   request_id=<id> intent=execution confidence=0.95 duration_ms=312
event=router_selected       request_id=<id> intent=execution handler=worker
event=worker_start          request_id=<id> worker=worker_llm intent=execution
event=worker_complete       request_id=<id> duration_ms=1450
event=agent_output_received request_id=<id>
event=tool_execute          tool=read_file request_id=<id>
event=tool_execute_complete tool=read_file request_id=<id>
event=request_complete      request_id=<id> intent=execution confidence=0.95 handler=worker total_latency_ms=1765
```

### Trace a specific request

```bash
docker compose logs ingress | grep "request_id=<paste-id-here>"
```

---

## 7. Verify the health endpoint

```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

---

## 8. Debugging

### Tests fail to import modules

Ensure you are running pytest from the project root:

```bash
cd /path/to/COREtex
pytest tests/test_smoke.py -v
```

The `pytest.ini` file sets `pythonpath = .` so all imports resolve correctly.

### Ollama errors in logs

If you see `event=pipeline_classifier_failure` or `event=pipeline_worker_failure`, Ollama is not reachable. Check:

```bash
curl http://localhost:11434/api/tags
```

If using Docker, the ingress container reaches Ollama via `host.docker.internal:11434`. Ensure Ollama is running on the host and is not firewall-blocked.

### Model not found

```bash
ollama pull llama3.2:3b
```

Override the model via environment variable if needed:

```bash
WORKER_MODEL=llama3.1:8b docker compose up --build
```

### Tool is not executing

Check the logs for `event=agent_output_parse_error` or `event=pipeline_agent_parse_failure`. This means the LLM did not return valid JSON. Try a larger model, or test the tool layer directly using the Python REPL approach above.
