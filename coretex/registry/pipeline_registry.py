"""PipelineRegistry — registry for named execution pipelines."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from coretex.runtime.pipeline import PipelineDefinition

logger = logging.getLogger(__name__)


class PipelineRegistry:
    """Holds named PipelineDefinition objects registered at startup.

    Introduced in v0.4.0 as a fully validated registry for configurable pipelines.
    Stores PipelineDefinition objects by name, allowing distributions to register
    one or more named pipelines and look them up at runtime.

    All ``register`` calls raise ``ValueError`` on duplicate names.
    ``get`` raises ``ValueError`` on unknown names and emits a structured
    ``event=registry_lookup_failed`` log.
    """

    def __init__(self) -> None:
        self._pipelines: Dict[str, object] = {}

    def register(self, name: str, pipeline: object) -> None:
        """Register *pipeline* under *name*.

        Raises:
            ValueError: If a pipeline with the same name is already registered.
        """
        if name in self._pipelines:
            raise ValueError(f"Pipeline already registered: {name}")
        self._pipelines[name] = pipeline
        logger.info("event=pipeline_registered name=%s", name)

    def get(self, name: str) -> object:
        """Return the pipeline registered as *name*.

        Raises:
            ValueError: If no pipeline with that name has been registered,
                        and logs ``event=registry_lookup_failed``.
        """
        if name not in self._pipelines:
            logger.error("event=registry_lookup_failed component=pipeline name=%s", name)
            raise ValueError(f"Unknown pipeline: {name}")
        return self._pipelines[name]

    def list(self) -> List[str]:
        """Return a list of all registered pipeline names."""
        return list(self._pipelines.keys())
