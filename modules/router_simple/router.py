"""RouterSimple — deterministic intent→handler router.

This is the v0.3.0 successor to app/router.py, refactored as a module that
implements the Router interface.
"""

from __future__ import annotations

import logging

from coretex.interfaces.router import Router

logger = logging.getLogger(__name__)

ROUTES: dict[str, str] = {
    "execution": "worker",
    "planning": "worker",
    "analysis": "worker",
    "ambiguous": "clarify",
}


class RouterSimple(Router):
    """Deterministic router: pure dict lookup, no LLM, no probabilistic logic."""

    def route(self, intent: str, request_id: str = "", **kwargs: object) -> str:
        """Return the handler name for *intent*. Unknown intents map to 'clarify'."""
        handler = ROUTES.get(intent)
        if handler is None:
            logger.warning(
                "event=router_fallback request_id=%s intent=%r",
                request_id, intent,
            )
            handler = "clarify"

        user_input = kwargs.get("user_input", "")
        confidence = kwargs.get("confidence", 0.0)
        logger.info(
            "event=intent_router request_id=%s intent=%s route=%s confidence=%.2f input=%r",
            request_id, intent, handler, confidence,
            str(user_input)[:120] if user_input else "",
        )
        return handler
