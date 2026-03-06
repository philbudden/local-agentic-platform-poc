"""Deterministic router — maps classifier intent to execution path.

The router is NOT an LLM. No probabilistic decisions are made here.
"""

import logging

logger = logging.getLogger(__name__)

ROUTES: dict[str, str] = {
    "execution": "worker",
    "planning": "worker",
    "analysis": "worker",
    "ambiguous": "clarify",
}


def route(
    intent: str,
    request_id: str = "",
    user_input: str = "",
    confidence: float = 0.0,
) -> str:
    """Return the handler name for *intent*.

    Unknown intents are treated as ambiguous and mapped to 'clarify'.
    """
    handler = ROUTES.get(intent)
    if handler is None:
        logger.warning(
            "event=router_fallback request_id=%s intent=%r",
            request_id, intent,
        )
        handler = "clarify"

    logger.info(
        "event=intent_router request_id=%s intent=%s route=%s confidence=%.2f",
        request_id, intent, handler, confidence,
    )
    return handler
