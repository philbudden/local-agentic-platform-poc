"""Classifier interface — contract for intent classification modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ClassificationResult:
    """The output of a classifier: an intent label, confidence score, and source."""

    intent: str
    confidence: float
    source: str = "llm"


class Classifier(ABC):
    """Abstract base class for classifier modules."""

    @abstractmethod
    async def classify(self, user_input: str, request_id: str = "") -> ClassificationResult:
        """Classify *user_input* and return a ClassificationResult."""
        ...
