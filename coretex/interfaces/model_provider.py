"""ModelProvider interface — contract for model inference backend modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class ModelProvider(ABC):
    """Abstract base class for model provider modules.

    Implementations wrap a specific inference backend (e.g. Ollama, OpenAI).
    """

    @abstractmethod
    async def generate(self, model: str, prompt: str, **kwargs: object) -> str:
        """Run a single-turn generation and return the response text."""
        ...

    @abstractmethod
    async def chat(self, model: str, messages: List[Dict[str, str]], **kwargs: object) -> str:
        """Run a multi-turn chat completion and return the assistant message text."""
        ...
