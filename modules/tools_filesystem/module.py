"""tools_filesystem — registration entrypoint.

Registers the read_file tool with the tool registry.
"""

from __future__ import annotations

from coretex.registry.model_registry import ModelProviderRegistry
from coretex.registry.module_registry import ModuleRegistry
from coretex.registry.tool_registry import ToolRegistry
from modules.tools_filesystem.filesystem import read_file


def register(
    module_registry: ModuleRegistry,
    tool_registry: ToolRegistry,
    model_registry: ModelProviderRegistry,
) -> None:
    """Register filesystem tools."""
    tool_registry.register(
        name="read_file",
        description="Read the text content of a local file",
        input_schema={"path": "string"},
        function=read_file,
    )
