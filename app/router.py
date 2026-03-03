"""Deterministic router — maps classifier intent to execution path.

The router is NOT an LLM. No probabilistic decisions are made here.
"""

ROUTES: dict[str, str] = {
    "execution": "worker",
    "decomposition": "worker",
    "novel_reasoning": "worker",
    "ambiguous": "clarify",
}


def route(intent: str) -> str:
    """Return the handler name for *intent*.

    Unknown intents are treated as ambiguous and mapped to 'clarify'.
    """
    return ROUTES.get(intent, "clarify")
