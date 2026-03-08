"""OllamaProvider — model provider implementation for the Ollama inference backend."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from coretex.config.settings import settings
from coretex.interfaces.model_provider import ModelProvider

logger = logging.getLogger(__name__)


class OllamaProvider(ModelProvider):
    """Wraps the Ollama HTTP API for both generation and chat endpoints."""

    async def generate(self, model: str, prompt: str, **kwargs: Any) -> str:
        """Call Ollama /api/generate and return the response text."""
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": kwargs.get("num_predict", settings.max_tokens),
            },
        }
        timeout = kwargs.get("timeout", settings.worker_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()["response"]

    async def chat(self, model: str, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Call Ollama /api/chat and return the assistant message text."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if "format" in kwargs:
            payload["format"] = kwargs["format"]
        if "options" in kwargs:
            payload["options"] = kwargs["options"]

        timeout = kwargs.get("timeout", settings.classifier_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
