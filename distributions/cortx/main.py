"""Ingress API — FastAPI entry point for the cortx distribution.

Exposes:
  POST /ingest                  — internal schema used by custom clients
  GET  /v1/models               — OpenAI-compatible model list for OpenWebUI
  POST /v1/chat/completions     — OpenAI-compatible shim for OpenWebUI
  GET  /debug/routes            — development: inspect intent→handler routing table
  GET  /health                  — health check
"""

import logging
import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from coretex.config.settings import settings
from coretex.runtime.context import ExecutionContext
from coretex.runtime.pipeline import PipelineRunner
from distributions.cortx.bootstrap import module_registry, tool_registry
from distributions.cortx.models import IngestRequest, IngestResponse
from modules.router_simple.router import ROUTES

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="CortX — cortx distribution")

pipeline = PipelineRunner(module_registry=module_registry, tool_registry=tool_registry)

_CLARIFY_RESPONSE = (
    "I'm not sure what you're asking. Could you provide more detail or clarify your request?"
)


# ---------------------------------------------------------------------------
# Internal endpoint
# ---------------------------------------------------------------------------


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Accept user input, run the pipeline, and return a structured response."""
    context = ExecutionContext(user_input=request.input)

    response_text, intent, confidence = await pipeline.run(context)

    return IngestResponse(intent=intent, confidence=confidence, response=response_text)


# ---------------------------------------------------------------------------
# OpenAI-compatible shim so OpenWebUI can treat this service as an LLM backend
# ---------------------------------------------------------------------------


class _OAIMessage(BaseModel):
    role: str
    content: str


class _OAIChatRequest(BaseModel):
    model: str = "agentic"
    messages: list[_OAIMessage]
    stream: bool = False


@app.get("/v1/models")
async def list_models() -> dict:
    """Return a minimal OpenAI-compatible model list so OpenWebUI can populate its dropdown."""
    return {
        "object": "list",
        "data": [
            {
                "id": "agentic",
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: _OAIChatRequest) -> dict:
    """Translate an OpenAI chat-completions request into an /ingest call."""
    user_text = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        "",
    )

    if not user_text.strip():
        result = IngestResponse(intent="ambiguous", confidence=0.0, response=_CLARIFY_RESPONSE)
    else:
        result = await ingest(IngestRequest(input=user_text))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.response},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Debug endpoint (development only)
# ---------------------------------------------------------------------------


@app.get("/debug/routes")
async def debug_routes() -> dict:
    """Return the current intent→handler routing table for development inspection."""
    return {"routes": ROUTES}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
