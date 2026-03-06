"""Ingress API — entry point for all user requests.

Exposes:
  POST /ingest                  — internal schema used by custom clients
  POST /v1/chat/completions     — OpenAI-compatible shim for OpenWebUI
  GET  /debug/routes            — development: inspect intent→handler routing table
"""

import logging
import time
import uuid

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from app.classifier import classify
from app.models import IngestRequest, IngestResponse
from app.router import ROUTES, route
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
    request_id = uuid.uuid4().hex
    t_start = time.monotonic()

    logger.info(
        "event=request_received request_id=%s",
        request_id,
    )

    classifier_result = await classify(request.input, request_id)
    t_classified = time.monotonic()

    handler = route(
        classifier_result.intent,
        request_id=request_id,
        user_input=request.input,
        confidence=classifier_result.confidence,
    )

    if handler == "clarify":
        response_text = _CLARIFY_RESPONSE
        t_worker = t_classified
    else:
        try:
            response_text = await generate(request.input, classifier_result.intent, request_id)
        except httpx.HTTPError as exc:
            status = getattr(exc.response, "status_code", "N/A") if hasattr(exc, "response") else "N/A"
            body = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    body = exc.response.text[:200]
                except Exception:
                    pass
            logger.error(
                "event=worker_error request_id=%s error_type=%s status=%s body=%r error=%s",
                request_id, type(exc).__name__, status, body, exc,
            )
            response_text = _WORKER_FAILURE_RESPONSE
            classifier_result = classifier_result.model_copy(
                update={"intent": "ambiguous", "confidence": 0.0}
            )
        t_worker = time.monotonic()

    total_latency_ms = int((time.monotonic() - t_start) * 1000)
    logger.info(
        "event=request_complete request_id=%s intent=%s confidence=%.2f "
        "classifier_latency_ms=%d worker_latency_ms=%d total_latency_ms=%d",
        request_id,
        classifier_result.intent,
        classifier_result.confidence,
        int((t_classified - t_start) * 1000),
        int((t_worker - t_classified) * 1000),
        total_latency_ms,
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
