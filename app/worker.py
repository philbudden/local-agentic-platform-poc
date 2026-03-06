"""Worker agent — calls the LLM to produce the final user-facing response.

Receives original user input and classifier intent; returns free-form text.
No memory, no tools.
"""

import logging
import time

import httpx

from app.settings import settings

logger = logging.getLogger(__name__)

# Intent-specific prompt templates; user input is appended via concatenation to
# avoid Python str.format() treating user-supplied braces as placeholders.
#
# All prompts instruct the LLM to respond with a JSON action envelope so that
# the ToolExecutor can either relay the content directly or invoke a tool.
_PROMPTS: dict[str, str] = {
    "execution": (
        "You are a precise assistant. The user has asked you to perform a concrete task.\n"
        "Give a direct, concise answer. No planning structure. No preamble. No commentary.\n"
        "Respond in no more than 150 words.\n\n"
        'You MUST respond with valid JSON using exactly this format:\n'
        '{"action": "respond", "content": "<your complete answer here>"}\n\n'
        "User request: "
    ),
    "planning": (
        "You are a structured assistant. The user needs a task broken into steps.\n"
        "Provide exactly 3 to 5 numbered steps. One sentence per step. No preamble.\n\n"
        'You MUST respond with valid JSON using exactly this format:\n'
        '{"action": "respond", "content": "<your numbered steps here>"}\n\n'
        "User request: "
    ),
    "analysis": (
        "You are an analytical assistant. The user wants open-ended thinking.\n"
        "Give a focused, insightful response. Limit to 3 sentences.\n\n"
        'You MUST respond with valid JSON using exactly this format:\n'
        '{"action": "respond", "content": "<your analytical response here>"}\n\n'
        "User request: "
    ),
}

_FALLBACK_PROMPT = _PROMPTS["execution"]


async def generate(user_input: str, intent: str, request_id: str = "") -> str:
    """Generate a response for *user_input* given *intent*.

    Raises httpx.HTTPError on network or HTTP failures (caller handles gracefully).
    """
    prompt = _PROMPTS.get(intent, _FALLBACK_PROMPT) + user_input

    if settings.debug_router:
        logger.debug(
            "event=worker_prompt request_id=%s prompt=%r",
            request_id, prompt[:500],
        )

    payload = {
        "model": settings.worker_model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": settings.max_tokens},
    }
    logger.info(
        "event=worker_start request_id=%s worker=worker intent=%s",
        request_id, intent,
    )
    logger.info(
        "event=llm_call request_id=%s call=2/2 model=%s intent=%s",
        request_id, settings.worker_model, intent,
    )
    t_start = time.monotonic()
    async with httpx.AsyncClient(timeout=settings.worker_timeout) as client:
        resp = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
        resp.raise_for_status()
        response_text = resp.json()["response"]
    latency_ms = int((time.monotonic() - t_start) * 1000)
    logger.info(
        "event=worker_complete request_id=%s worker=worker latency_ms=%d",
        request_id, latency_ms,
    )
    return response_text
