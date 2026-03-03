"""Ingress API — entry point for all user requests.

Exposes:
  POST /ingest                  — internal schema used by custom clients
  POST /v1/chat/completions     — OpenAI-compatible shim for OpenWebUI
"""

import logging
import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from app.classifier import classify
from app.models import IngestRequest, IngestResponse
from app.router import route
from app.worker import generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Platform — Ingress API")

_CLARIFY_RESPONSE = (
    "I'm not sure what you're asking. Could you provide more detail or clarify your request?"
)


# ---------------------------------------------------------------------------
# Internal endpoint
# ---------------------------------------------------------------------------


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Accept user input, orchestrate classification and routing, return response."""
    t_start = time.monotonic()

    classifier_result = await classify(request.input)
    t_classified = time.monotonic()

    handler = route(classifier_result.intent)

    if handler == "clarify":
        response_text = _CLARIFY_RESPONSE
        t_worker = t_classified
    else:
        response_text = await generate(request.input, classifier_result.intent)
        t_worker = time.monotonic()

    total_latency = time.monotonic() - t_start
    logger.info(
        "ingest completed",
        extra={
            "intent": classifier_result.intent,
            "confidence": classifier_result.confidence,
            "classifier_latency_s": round(t_classified - t_start, 3),
            "worker_latency_s": round(t_worker - t_classified, 3),
            "total_latency_s": round(total_latency, 3),
        },
    )

    return IngestResponse(
        intent=classifier_result.intent,
        confidence=classifier_result.confidence,
        response=response_text,
    )


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


@app.post("/v1/chat/completions")
async def chat_completions(request: _OAIChatRequest) -> dict:
    """Translate an OpenAI chat-completions request into an /ingest call.

    Extracts the last user message, forwards it to the ingest logic, and wraps
    the result in a minimal ChatCompletion-shaped response.
    """
    user_text = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        "",
    )

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
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
