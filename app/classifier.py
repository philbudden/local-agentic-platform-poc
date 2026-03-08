"""Classifier agent — calls the LLM to categorise user intent.

Returns strict JSON: {"intent": <str>, "confidence": <float>}

Uses the Ollama /api/chat endpoint with a system message so that
instruction-tuned models follow the schema reliably.

Retries once if the response is not valid JSON; falls back to
intent="ambiguous", confidence=0.0 on second failure.
"""

import json
import logging
import time
from typing import Optional

import httpx
from pydantic import ValidationError

from app.models import ClassifierResponse
from app.settings import settings

logger = logging.getLogger(__name__)

# System message sent to the model as the authoritative instruction.
# User input is passed as a separate user turn — never interpolated here.
_SYSTEM_PROMPT = """\
You are a strict intent classifier.

Respond with ONLY valid JSON:
{"intent": "<category>", "confidence": <0.0-1.0>}

Categories:
- execution
- planning
- analysis
- ambiguous

Definitions:

execution:
The user asks to create, write, generate, compose, draft, produce, summarise, translate, calculate, code, or output something.
Creativity does NOT matter. If something must be produced, it is execution.

planning:
The user asks how to do something or asks for steps, a plan, a process, or how something should be built.
Keywords often include: how, steps, build, launch, start, implement.

analysis:
The user asks for design, evaluation, comparison, implications, or open-ended thinking.
Keywords often include: design, compare, analyse, evaluate, implications.

ambiguous:
The request is a greeting, fragment, or unclear.

Important:
If the request asks to write, generate, or create something, it is ALWAYS execution.
If none of the definitions clearly match, choose ambiguous.

Examples:

User: Write a haiku about rain.
{"intent": "execution", "confidence": 0.95}

User: Generate a short sci-fi story.
{"intent": "execution", "confidence": 0.95}

User: Summarise this in 3 sentences.
{"intent": "execution", "confidence": 0.95}

User: How do I start a podcast?
{"intent": "planning", "confidence": 0.9}

User: Compare Kubernetes and Nomad.
{"intent": "analysis", "confidence": 0.9}

User: Hello
{"intent": "ambiguous", "confidence": 0.9}

User: How would I build a scalable SaaS architecture?
{"intent": "planning", "confidence": 0.9}

User: What steps are required to launch a startup?
{"intent": "planning", "confidence": 0.9}

User: Design a new economic system for Mars colonies.
{"intent": "analysis", "confidence": 0.9}

User: What are the implications of AI replacing software engineers?
{"intent": "analysis", "confidence": 0.9}
"""

# Maps common LLM-generated intent variants to valid schema values.
_INTENT_ALIASES: dict[str, str] = {
    "creative_writing": "execution",
    "creative": "execution",
    "generation": "execution",
    "task": "execution",
    "action": "execution",
    "command": "execution",
    "novel_reasoning": "analysis",
    "reasoning": "analysis",
    "explanation": "analysis",
    "synthesis": "analysis",
    "decomposition": "planning",
    "complex": "planning",
    "unclear": "ambiguous",
    "unknown": "ambiguous",
    "other": "ambiguous",
}

# Alternative field names some models use instead of "intent".
_INTENT_FIELD_CANDIDATES = ("intent", "category", "type", "classification", "class")


_EXECUTION_PREFIXES = (
    "write", "generate", "create", "compose",
    "draft", "produce", "summarise",
    "summarize", "translate", "calculate",
    "code", "list",
)

_PLANNING_PREFIXES = (
    "how do i",
    "how would i",
    "how can i",
    "what steps",
)

_AMBIGUOUS_SHORT = (
    "help",
    "help.",
    "hi",
    "hello",
    "hey",
    "ok",
    "okay",
    "thanks",
)

async def classify(user_input: str, request_id: str = "") -> ClassifierResponse:
    """Classify *user_input* and return a ClassifierResponse.

    Applies a deterministic prefix check before calling the LLM to eliminate
    the most common misclassifications. Retries once on bad JSON or network
    error; returns fallback on second failure.
    """
    t_start = time.monotonic()
    lower = user_input.lower().strip()

    if lower.startswith(_EXECUTION_PREFIXES):
        result = ClassifierResponse(intent="execution", confidence=0.95)
        logger.info(
            "event=classifier_result request_id=%s intent=%s confidence=%.2f source=prefix_match",
            request_id, result.intent, result.confidence,
        )
        return result

    if lower.startswith(_PLANNING_PREFIXES):
        result = ClassifierResponse(intent="planning", confidence=0.95)
        logger.info(
            "event=classifier_result request_id=%s intent=%s confidence=%.2f source=prefix_match",
            request_id, result.intent, result.confidence,
        )
        return result

    if lower.startswith(_AMBIGUOUS_SHORT):
        result = ClassifierResponse(intent="ambiguous", confidence=0.95)
        logger.info(
            "event=classifier_result request_id=%s intent=%s confidence=%.2f source=prefix_match",
            request_id, result.intent, result.confidence,
        )
        return result

    for attempt in range(2):
        try:
            raw = await _call_ollama(user_input, request_id)
        except (httpx.HTTPError, httpx.RequestError) as exc:
            body = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    body = exc.response.text[:200]
                except Exception:
                    pass
            logger.warning(
                "event=classifier_retry request_id=%s attempt=%d reason=network_error "
                "error_type=%s body=%r error=%r",
                request_id, attempt + 1, type(exc).__name__, body, str(exc) or repr(exc),
            )
            continue
        result = _parse(raw)
        if result is not None:
            latency_ms = int((time.monotonic() - t_start) * 1000)
            logger.info(
                "event=classifier_result request_id=%s intent=%s confidence=%.2f source=llm",
                request_id, result.intent, result.confidence,
            )
            logger.info(
                "event=classifier_latency request_id=%s latency_ms=%d",
                request_id, latency_ms,
            )
            return result
        logger.warning(
            "event=classifier_retry request_id=%s attempt=%d reason=invalid_json raw=%r",
            request_id, attempt + 1, raw,
        )

    latency_ms = int((time.monotonic() - t_start) * 1000)
    logger.error(
        "event=classifier_fallback request_id=%s reason=max_retries_exceeded latency_ms=%d",
        request_id, latency_ms,
    )
    return ClassifierResponse(intent="ambiguous", confidence=0.0)


async def _call_ollama(user_input: str, request_id: str = "") -> str:
    """Call Ollama /api/chat with a system message + user turn."""
    if settings.debug_router:
        logger.debug(
            "event=classifier_prompt request_id=%s prompt=%r",
            request_id, _SYSTEM_PROMPT[:500],
        )
    payload = {
        "model": settings.classifier_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "top_p": 0.8, "num_predict": 32},
    }
    logger.info(
        "event=llm_call request_id=%s call=1/2 model=%s",
        request_id, settings.classifier_model,
    )
    async with httpx.AsyncClient(timeout=settings.classifier_timeout) as client:
        resp = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        logger.debug(
            "event=classifier_raw_output request_id=%s output=%r",
            request_id, raw,
        )
        return raw


def _parse(raw: str) -> Optional[ClassifierResponse]:
    text = raw.strip()
    # Strip markdown code fences in case a model ignores the format constraint.
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    # Try strict parse first.
    try:
        return ClassifierResponse(**data)
    except ValidationError:
        pass

    # Normalise: find the intent value under any common field name, then map aliases.
    raw_intent = None
    for field in _INTENT_FIELD_CANDIDATES:
        if field in data:
            raw_intent = str(data[field]).strip().lower().replace(" ", "_").replace("-", "_")
            break

    if raw_intent is None:
        logger.warning("Classifier response has no recognisable intent field: %r", data)
        return None

    intent = _INTENT_ALIASES.get(raw_intent, raw_intent)
    confidence = float(data.get("confidence", data.get("score", data.get("certainty", 0.5))))

    try:
        return ClassifierResponse(intent=intent, confidence=confidence)
    except ValidationError:
        logger.warning(
            "Classifier intent %r (normalised from %r) not in schema; treating as ambiguous",
            intent, raw_intent,
        )
        return ClassifierResponse(intent="ambiguous", confidence=0.0)

