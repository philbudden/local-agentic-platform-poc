"""ModelProviderRegistry — registry for model inference backend providers."""

from __future__ import annotations

import logging
from typing import Dict

from cortx.interfaces.model_provider import ModelProvider

logger = logging.getLogger(__name__)


class ModelProviderRegistry:
    """Holds model providers registered by modules at startup."""

    def __init__(self) -> None:
        self._providers: Dict[str, ModelProvider] = {}

    def register(self, name: str, provider: ModelProvider) -> None:
        if name in self._providers:
            raise ValueError(f"ModelProvider already registered: {name}")
        self._providers[name] = provider
        logger.info("event=model_provider_registered name=%s", name)

    def get(self, name: str) -> ModelProvider:
        if name not in self._providers:
            raise ValueError(f"Unknown model provider: {name}")
        return self._providers[name]

    def list(self) -> list[str]:
        return list(self._providers.keys())
