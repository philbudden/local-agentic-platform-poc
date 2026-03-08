"""Runtime executor — agent action model, tool executor, and output parser.

This module is the v0.3.0 successor to core/tools.py.

Design rules (unchanged from v0.2.0):
  - Agents never execute tools directly — only ToolExecutor can run tools.
  - Agent output must be strict JSON; parse_agent_output validates it.
  - All steps emit structured log events for full observability.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from coretex.registry.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent Action Model
# ---------------------------------------------------------------------------


class AgentAction:
    def __init__(
        self,
        action: Optional[str],
        tool: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
    ) -> None:
        self.action = action
        self.tool = tool
        self.args = args or {}
        self.content = content

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentAction:
        logger.info(
            "event=agent_action_parsed action=%s tool=%s",
            data.get("action"),
            data.get("tool"),
        )
        return cls(
            action=data.get("action"),
            tool=data.get("tool"),
            args=data.get("args"),
            content=data.get("content"),
        )


# ---------------------------------------------------------------------------
# Tool Executor
# ---------------------------------------------------------------------------


class ToolExecutor:
    """The only component that can run tools. Dispatches on AgentAction.action."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, action: AgentAction, request_id: str = "") -> Any:
        logger.info(
            "event=executor_received action=%s tool=%s request_id=%s",
            action.action,
            action.tool,
            request_id,
        )

        if action.action == "respond":
            logger.info("event=executor_direct_response request_id=%s", request_id)
            return action.content

        if action.action == "tool":
            if not action.tool:
                logger.error(
                    "event=executor_tool_name_missing request_id=%s",
                    request_id,
                )
                raise ValueError("Tool action is missing a tool name")

            tool = self.registry.get(action.tool)
            result = tool.execute(action.args, request_id=request_id)

            logger.info(
                "event=tool_result tool=%s request_id=%s result_type=%s",
                action.tool,
                request_id,
                type(result).__name__,
            )

            return result

        logger.error("event=executor_unknown_action action=%s request_id=%s", action.action, request_id)
        raise ValueError(f"Unknown action type: {action.action}")


# ---------------------------------------------------------------------------
# Agent Output Parsing
# ---------------------------------------------------------------------------


def parse_agent_output(raw: str, request_id: str = "") -> AgentAction:
    """Parse a JSON string emitted by the agent into an AgentAction.

    Raises json.JSONDecodeError if *raw* is not valid JSON, or any other
    exception if the parsed structure is unusable.  The caller is responsible
    for graceful fallback.
    """
    logger.info(
        "event=agent_output_received request_id=%s raw=%r",
        request_id,
        raw[:200] if raw else "",
    )

    try:
        data = json.loads(raw)
        return AgentAction.from_dict(data)
    except Exception as exc:
        logger.error(
            "event=agent_output_parse_error request_id=%s error=%r raw=%r",
            request_id,
            str(exc),
            raw[:200],
        )
        raise
