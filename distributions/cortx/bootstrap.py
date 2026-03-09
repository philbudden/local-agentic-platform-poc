"""Bootstrap — load all modules and build the shared registries for cortx."""

from __future__ import annotations

from coretex.registry.model_registry import ModelProviderRegistry
from coretex.registry.module_registry import ModuleRegistry
from coretex.registry.tool_registry import ToolRegistry
from coretex.runtime.loader import ModuleLoader

# ---------------------------------------------------------------------------
# Registries (singletons shared across the application)
# ---------------------------------------------------------------------------

module_registry = ModuleRegistry()
tool_registry = ToolRegistry()
model_registry = ModelProviderRegistry()

# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------

_loader = ModuleLoader(
    module_registry=module_registry,
    tool_registry=tool_registry,
    model_registry=model_registry,
)

_loader.load_all([
    "modules.model_provider_ollama.module",
    "modules.classifier_basic.module",
    "modules.router_simple.module",
    "modules.worker_llm.module",
    "modules.tools_filesystem.module",
])
