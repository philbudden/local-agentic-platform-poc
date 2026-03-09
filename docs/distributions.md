# Distributions Guide

> Version: v0.3.x Stabilisation

---

## Overview

A **distribution** is a complete, runnable CortX system assembled from:

- the COREtex runtime
- a set of modules
- infrastructure (e.g. FastAPI, Docker)

Distributions are the user-facing product layer. The runtime and modules remain independent.

The first distribution is `cortx`, located in `distributions/cortx/`.

---

## How Bootstrap Works

Every distribution must bootstrap the runtime before serving requests. Bootstrap performs three steps:

1. **Create the registries** — instantiate `ModuleRegistry`, `ToolRegistry`, and `ModelProviderRegistry`.
2. **Create the ModuleLoader** — wire the registries into a loader.
3. **Load modules** — call `loader.load_all([...])` with the list of module paths.

After bootstrap completes, the registries hold all registered components and the `PipelineRunner` can be created.

### Example Bootstrap (`distributions/cortx/bootstrap.py`)

```python
from coretex.registry.model_registry import ModelProviderRegistry
from coretex.registry.module_registry import ModuleRegistry
from coretex.registry.tool_registry import ToolRegistry
from coretex.runtime.loader import ModuleLoader

module_registry = ModuleRegistry()
tool_registry = ToolRegistry()
model_registry = ModelProviderRegistry()

loader = ModuleLoader(
    module_registry=module_registry,
    tool_registry=tool_registry,
    model_registry=model_registry,
)

loader.load_all([
    "modules.model_provider_ollama.module",
    "modules.classifier_basic.module",
    "modules.router_simple.module",
    "modules.worker_llm.module",
    "modules.tools_filesystem.module",
])
```

`load_all()` emits structured lifecycle logs:

```
event=module_loading_start count=5 modules=[...]
event=module_loaded module=modules.classifier_basic.module registered_components=1
event=module_loaded module=modules.tools_filesystem.module registered_components=1
event=module_loading_complete loaded=5
```

---

## How to Build a Distribution

### 1. Create the distribution directory

```
distributions/
  my_distribution/
    __init__.py
    bootstrap.py   ← module loading
    main.py        ← application entrypoint (e.g. FastAPI)
```

### 2. Write `bootstrap.py`

Follow the bootstrap pattern above. Choose which modules to load.

### 3. Write `main.py`

Create the application ingress. For a FastAPI distribution:

```python
from fastapi import FastAPI
from coretex.runtime.context import ExecutionContext
from coretex.runtime.pipeline import PipelineRunner
from distributions.my_distribution.bootstrap import module_registry, tool_registry

app = FastAPI()

pipeline = PipelineRunner(module_registry, tool_registry)


@app.post("/ingest")
async def ingest(request: IngestRequest):
    ctx = ExecutionContext(user_input=request.input)
    response, intent, confidence = await pipeline.run(ctx)
    return {"response": response, "intent": intent, "confidence": confidence}
```

### 4. Add infrastructure

Distributions may include:

- `Dockerfile` for containerisation
- `docker-compose.yml` for multi-service orchestration
- `.env` for local configuration
- Any infrastructure-specific code

---

## The `cortx` Distribution

The `cortx` distribution is the reference implementation. It provides:

| Component | Module |
|-----------|--------|
| Classifier | `modules.classifier_basic.module` |
| Router | `modules.router_simple.module` |
| Worker | `modules.worker_llm.module` |
| Filesystem tool | `modules.tools_filesystem.module` |
| Ollama model provider | `modules.model_provider_ollama.module` |
| FastAPI ingress | `distributions/cortx/main.py` |
| OpenAI chat shim | `/v1/chat/completions` endpoint |

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Main request pipeline |
| `/v1/chat/completions` | POST | OpenAI-compatible shim for OpenWebUI |
| `/v1/models` | GET | Model list (OpenAI-compatible) |
| `/health` | GET | Health check |
| `/debug/routes` | GET | Current routing table |

---

## Distribution vs Runtime

| Runtime (`coretex/`) | Distribution (`distributions/`) |
|----------------------|----------------------------------|
| Platform primitives | Assembled application |
| No infrastructure deps | May depend on FastAPI, Docker, etc. |
| No module imports | Loads and wires modules |
| Stable across releases | May change per deployment need |
| Tested in isolation | Tested end-to-end |

---

## Docker Deployment

The `cortx` distribution includes a `Dockerfile` and `docker-compose.yml`.

```bash
# Start the full stack (COREtex + OpenWebUI)
docker-compose up -d

# COREtex API: http://localhost:8000
# OpenWebUI:   http://localhost:3000
```

Ollama runs on the host machine. The Docker container reaches it via `host.docker.internal:11434`.

See `README.md` for full deployment instructions.
