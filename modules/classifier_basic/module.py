"""classifier_basic — registration entrypoint.

Registers the ClassifierBasic instance with the module registry.
"""

from __future__ import annotations

from coretex.registry.model_registry import ModelProviderRegistry
from coretex.registry.module_registry import ModuleRegistry
from coretex.registry.tool_registry import ToolRegistry
from modules.classifier_basic.classifier import ClassifierBasic


def register(
    module_registry: ModuleRegistry,
    tool_registry: ToolRegistry,
    model_registry: ModelProviderRegistry,
) -> None:
    """Register the basic classifier."""
    module_registry.register_classifier("classifier_basic", ClassifierBasic())
