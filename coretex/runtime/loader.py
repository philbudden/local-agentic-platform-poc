"""ModuleLoader — loads modules by dotted path and calls their register() function."""

from __future__ import annotations

import importlib
import logging
from typing import Optional

from coretex.registry.model_registry import ModelProviderRegistry
from coretex.registry.module_registry import ModuleRegistry
from coretex.registry.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ModuleLoader:
    """Loads modules dynamically and registers their capabilities with the runtime.

    Each module must expose a top-level ``register()`` function with the signature::

        def register(
            module_registry: ModuleRegistry,
            tool_registry: ToolRegistry,
            model_registry: ModelProviderRegistry,
        ) -> None: ...

    Modules may ignore registries they do not need.
    """

    def __init__(
        self,
        module_registry: ModuleRegistry,
        tool_registry: ToolRegistry,
        model_registry: Optional[ModelProviderRegistry] = None,
    ) -> None:
        self._module_registry = module_registry
        self._tool_registry = tool_registry
        self._model_registry = model_registry or ModelProviderRegistry()

    def load(self, module_path: str) -> None:
        """Import *module_path* and call its ``register()`` function.

        Marks the module as loaded in the ModuleRegistry on success.
        Raises on any import or registration failure.
        """
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            logger.error("event=module_import_failed module=%s error=%r", module_path, str(exc))
            raise

        if not hasattr(mod, "register"):
            raise ValueError(f"Module {module_path!r} has no register() function")

        mod.register(
            module_registry=self._module_registry,
            tool_registry=self._tool_registry,
            model_registry=self._model_registry,
        )

        self._module_registry.mark_loaded(module_path)
        logger.info("event=module_loaded module=%s", module_path)
