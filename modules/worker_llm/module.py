"""worker_llm — registration entrypoint."""

from __future__ import annotations

from coretex.registry.model_registry import ModelProviderRegistry
from coretex.registry.module_registry import ModuleRegistry
from coretex.registry.tool_registry import ToolRegistry
from modules.worker_llm.worker import WorkerLLM


def register(
    module_registry: ModuleRegistry,
    tool_registry: ToolRegistry,
    model_registry: ModelProviderRegistry,
) -> None:
    """Register the LLM worker."""
    module_registry.register_worker("worker_llm", WorkerLLM())
