"""Router interface — contract for deterministic routing modules."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Router(ABC):
    """Abstract base class for router modules."""

    @abstractmethod
    def route(self, intent: str, request_id: str = "", **kwargs: object) -> str:
        """Map *intent* to a handler name and return it."""
        ...
