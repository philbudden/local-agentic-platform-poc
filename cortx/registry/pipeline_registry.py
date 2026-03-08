"""PipelineRegistry — registry for named execution pipelines."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PipelineRegistry:
    """Holds named pipelines registered at startup.

    In v0.3.0 this registry is a placeholder foundation for configurable pipelines
    (introduced in v0.4.0). It stores arbitrary pipeline objects by name.
    """

    def __init__(self) -> None:
        self._pipelines: Dict[str, Any] = {}

    def register(self, name: str, pipeline: Any) -> None:
        if name in self._pipelines:
            raise ValueError(f"Pipeline already registered: {name}")
        self._pipelines[name] = pipeline
        logger.info("event=pipeline_registered name=%s", name)

    def get(self, name: str) -> Any:
        if name not in self._pipelines:
            raise ValueError(f"Unknown pipeline: {name}")
        return self._pipelines[name]

    def list(self) -> list[str]:
        return list(self._pipelines.keys())
