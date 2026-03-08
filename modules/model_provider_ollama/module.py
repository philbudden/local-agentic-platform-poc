"""model_provider_ollama — registration entrypoint.

Registers the Ollama model provider with the runtime's model registry.
"""

from __future__ import annotations

from coretex.registry.model_registry import ModelProviderRegistry
from coretex.registry.module_registry import ModuleRegistry
from coretex.registry.tool_registry import ToolRegistry
from modules.model_provider_ollama.provider import OllamaProvider


def register(
    module_registry: ModuleRegistry,
    tool_registry: ToolRegistry,
    model_registry: ModelProviderRegistry,
) -> None:
    """Register the Ollama model provider."""
    model_registry.register("ollama", OllamaProvider())
