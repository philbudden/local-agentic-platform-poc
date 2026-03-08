"""ModuleRegistry — central registry where modules register their capabilities."""

from __future__ import annotations

import logging
from typing import Dict

from coretex.interfaces.classifier import Classifier
from coretex.interfaces.router import Router
from coretex.interfaces.worker import Worker

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """Holds classifiers, routers, and workers registered by modules at startup."""

    def __init__(self) -> None:
        self._classifiers: Dict[str, Classifier] = {}
        self._routers: Dict[str, Router] = {}
        self._workers: Dict[str, Worker] = {}
        self._loaded: list[str] = []

    # ------------------------------------------------------------------
    # Classifiers
    # ------------------------------------------------------------------

    def register_classifier(self, name: str, classifier: Classifier) -> None:
        if name in self._classifiers:
            raise ValueError(f"Classifier already registered: {name}")
        self._classifiers[name] = classifier
        logger.info("event=classifier_registered name=%s", name)

    def get_classifier(self, name: str) -> Classifier:
        if name not in self._classifiers:
            raise ValueError(f"Unknown classifier: {name}")
        return self._classifiers[name]

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------

    def register_router(self, name: str, router: Router) -> None:
        if name in self._routers:
            raise ValueError(f"Router already registered: {name}")
        self._routers[name] = router
        logger.info("event=router_registered name=%s", name)

    def get_router(self, name: str) -> Router:
        if name not in self._routers:
            raise ValueError(f"Unknown router: {name}")
        return self._routers[name]

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def register_worker(self, name: str, worker: Worker) -> None:
        if name in self._workers:
            raise ValueError(f"Worker already registered: {name}")
        self._workers[name] = worker
        logger.info("event=worker_registered name=%s", name)

    def get_worker(self, name: str) -> Worker:
        if name not in self._workers:
            raise ValueError(f"Unknown worker: {name}")
        return self._workers[name]

    # ------------------------------------------------------------------
    # Module tracking
    # ------------------------------------------------------------------

    def mark_loaded(self, module_path: str) -> None:
        self._loaded.append(module_path)

    def list_loaded(self) -> list[str]:
        return list(self._loaded)
