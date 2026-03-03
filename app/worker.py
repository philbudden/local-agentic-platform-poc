"""Worker agent — calls the LLM to produce the final user-facing response.

Receives original user input and classifier intent; returns free-form text.
No memory, no tools.
"""

import logging

import httpx

from app.settings import settings

logger = logging.getLogger(__name__)

# Prompt header is fixed text; user input is appended via concatenation to
# avoid Python str.format() treating user-supplied braces as placeholders.
_PROMPT_HEADER = "You are a helpful assistant. The user's request has been classified as: "
_PROMPT_BODY = "\n\nRespond helpfully and concisely to the following:\n"


async def generate(user_input: str, intent: str) -> str:
    """Generate a response for *user_input* given *intent*."""
    prompt = _PROMPT_HEADER + intent + _PROMPT_BODY + user_input
    payload = {
        "model": settings.worker_model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": settings.max_tokens},
    }
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        resp = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]
