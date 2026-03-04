"""Ingress API — entry point for all user requests.

Exposes:
  POST /ingest                  — internal schema used by custom clients
  POST /v1/chat/completions     — OpenAI-compatible shim for OpenWebUI
"""

import logging
import time
import uuid

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from app.classifier import classify
from app.models import IngestRequest, IngestResponse
from app.router import route
from app.settings import settings
from app.worker import generate

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Platform — Ingress API")

_CLARIFY_RESPONSE = (
    "I'm not sure what you're asking. Could you provide more detail or clarify your request?"
)
_WORKER_FAILURE_RESPONSE = (
    "I'm sorry, I was unable to process your request right now. Please try again later."
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
        try:
            response_text = await generate(request.input, classifier_result.intent)
        except httpx.HTTPError as exc:
            status = getattr(exc.response, "status_code", "N/A") if hasattr(exc, "response") else "N/A"
            logger.error(
                "Worker call failed: %s status=%s error=%s",
                type(exc).__name__, status, exc,
            )
            response_text = _WORKER_FAILURE_RESPONSE
            classifier_result = classifier_result.model_copy(
                update={"intent": "ambiguous", "confidence": 0.0}
            )
        t_worker = time.monotonic()

    total_latency = time.monotonic() - t_start
    logger.info(
        "request complete intent=%s confidence=%.2f "
        "classifier_latency_s=%.3f worker_latency_s=%.3f total_latency_s=%.3f",
        classifier_result.intent,
        classifier_result.confidence,
        round(t_classified - t_start, 3),
        round(t_worker - t_classified, 3),
        round(total_latency, 3),
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
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
