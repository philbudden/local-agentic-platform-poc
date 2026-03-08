"""WorkerLLM — response-generation worker using the Ollama inference backend.

This is the v0.3.0 successor to app/worker.py, refactored as a module that
implements the Worker interface.
"""

from __future__ import annotations

import logging
import time

import httpx

from coretex.config.settings import settings
from coretex.interfaces.worker import Worker

logger = logging.getLogger(__name__)

_PROMPTS: dict[str, str] = {
    "execution": (
        "You are a precise assistant. The user has asked you to perform a concrete task.\n"
        "Give a direct, concise answer. No planning structure. No preamble. No commentary.\n"
        "Respond in no more than 150 words.\n\n"
        "Respond with valid JSON in one of these two formats:\n"
        'Direct answer: {"action": "respond", "content": "<your complete answer here>"}\n'
        'Read a file:   {"action": "tool", "tool": "read_file", "args": {"path": "<absolute path>"}}\n\n'
        "Use the tool ONLY when the task explicitly requires reading file contents from disk.\n\n"
        "User request: "
    ),
    "planning": (
        "You are a structured assistant. The user needs a task broken into steps.\n"
        "Provide exactly 3 to 5 numbered steps. One sentence per step. No preamble.\n\n"
        "Respond with valid JSON in one of these two formats:\n"
        'Direct answer: {"action": "respond", "content": "<your numbered steps here>"}\n'
        'Read a file:   {"action": "tool", "tool": "read_file", "args": {"path": "<absolute path>"}}\n\n'
        "Use the tool ONLY when the task explicitly requires reading file contents from disk.\n\n"
        "User request: "
    ),
    "analysis": (
        "You are an analytical assistant. The user wants open-ended thinking.\n"
        "Give a focused, insightful response. Limit to 3 sentences.\n\n"
        "Respond with valid JSON in one of these two formats:\n"
        'Direct answer: {"action": "respond", "content": "<your analytical response here>"}\n'
        'Read a file:   {"action": "tool", "tool": "read_file", "args": {"path": "<absolute path>"}}\n\n'
        "Use the tool ONLY when the task explicitly requires reading file contents from disk.\n\n"
        "User request: "
    ),
}

_FALLBACK_PROMPT = _PROMPTS["execution"]


class WorkerLLM(Worker):
    """Worker that generates responses using the Ollama /api/generate endpoint."""

    async def generate(self, user_input: str, intent: str, request_id: str = "") -> str:
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
