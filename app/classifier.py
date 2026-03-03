"""Classifier agent — calls the LLM to categorise user intent.

Returns strict JSON: {"intent": <str>, "confidence": <float>}

Retries once if the response is not valid JSON; falls back to
intent="ambiguous", confidence=0.0 on second failure.
"""

import json
import logging

import httpx
from pydantic import ValidationError

from app.models import ClassifierResponse
from app.settings import settings

logger = logging.getLogger(__name__)

# Prompt header is fixed text; user input is appended via concatenation to
# avoid Python str.format() treating user-supplied braces as placeholders.
_PROMPT_HEADER = """\
Classify the following user input into exactly one intent category.

Reply with ONLY valid JSON — no prose, no markdown, no code fences.
Schema: {"intent": "<category>", "confidence": <0.0-1.0>}
Valid intents: execution, decomposition, novel_reasoning, ambiguous

User input: """


async def classify(user_input: str) -> ClassifierResponse:
    """Classify *user_input* and return a ClassifierResponse.

    Retries once on bad JSON or network error; returns fallback on second failure.
    """
    for attempt in range(2):
        try:
            raw = await _call_ollama(user_input)
        except httpx.HTTPError as exc:
            logger.warning("Classifier attempt %d network error: %s", attempt + 1, exc)
            continue
        result = _parse(raw)
        if result is not None:
            return result
        logger.warning("Classifier attempt %d produced invalid JSON: %r", attempt + 1, raw)

    logger.error("Classifier failed after 2 attempts; returning fallback.")
    return ClassifierResponse(intent="ambiguous", confidence=0.0)


async def _call_ollama(user_input: str) -> str:
    prompt = _PROMPT_HEADER + user_input
    payload = {
        "model": settings.classifier_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": settings.max_tokens},
    }
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        resp = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]


def _parse(raw: str) -> ClassifierResponse | None:
    try:
        data = json.loads(raw.strip())
        return ClassifierResponse(**data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None
