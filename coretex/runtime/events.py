"""EventBus — minimal synchronous event emission for observability.

All important runtime operations emit structured events through this bus.
In v0.3.0 this is a structured log wrapper. A full async subscriber/publisher
pattern will be introduced in v0.6.0.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EventBus:
    """Emits structured log events for runtime observability."""

    def emit(self, event_name: str, **kwargs: object) -> None:
        """Emit an INFO-level structured event."""
        parts = [f"event={event_name}"]
        for k, v in kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.2f}")
            elif isinstance(v, int):
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={v!r}")
        logger.info(" ".join(parts))

    def emit_warning(self, event_name: str, **kwargs: object) -> None:
        """Emit a WARNING-level structured event."""
        parts = [f"event={event_name}"]
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={v!r}")
        logger.warning(" ".join(parts))

    def emit_error(self, event_name: str, **kwargs: object) -> None:
        """Emit an ERROR-level structured event."""
        parts = [f"event={event_name}"]
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={v!r}")
        logger.error(" ".join(parts))


# Module-level singleton used by the runtime.
event_bus = EventBus()
