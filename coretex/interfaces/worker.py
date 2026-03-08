"""Worker interface — contract for response-generation modules."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Worker(ABC):
    """Abstract base class for worker modules."""

    @abstractmethod
    async def generate(self, user_input: str, intent: str, request_id: str = "") -> str:
        """Generate a response for *user_input* given *intent*.

        Must return a JSON action envelope string (or plain text as fallback).
        """
        ...
