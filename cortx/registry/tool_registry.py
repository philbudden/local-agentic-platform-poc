"""ToolRegistry — central registry for tools that can be invoked by ToolExecutor.

Moved from core/tools.py as part of the v0.3.0 runtime extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """A named, executable tool with metadata."""

    name: str
    description: str
    input_schema: Dict[str, str]
    function: Callable[..., Any]

    def execute(self, args: Dict[str, Any], request_id: str = "") -> Any:
        logger.info("event=tool_execute tool=%s request_id=%s args=%s", self.name, request_id, args)

        result = self.function(**args)

        logger.info(
            "event=tool_execute_complete tool=%s request_id=%s result_type=%s",
            self.name,
            request_id,
            type(result).__name__,
        )

        return result


class ToolRegistry:
    """Registry for tools callable by the ToolExecutor."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, str],
        function: Callable[..., Any],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")

        tool = Tool(
            name=name,
            description=description,
            input_schema=input_schema,
            function=function,
        )
        self._tools[name] = tool

        logger.info("event=tool_registered tool=%s schema=%s", name, input_schema)

    def get(self, name: str) -> Tool:
        logger.info("event=tool_lookup tool=%s", name)

        if name not in self._tools:
            logger.error("event=tool_lookup_failed tool=%s", name)
            raise ValueError(f"Unknown tool: {name}")

        return self._tools[name]

    def list(self) -> list:
        logger.debug("event=tool_list_requested count=%d", len(self._tools))
        return list(self._tools.keys())
