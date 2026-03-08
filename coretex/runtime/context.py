"""ExecutionContext — per-request context threaded through the pipeline."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionContext:
    """Carries all per-request state through the pipeline."""

    user_input: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    intent: Optional[str] = None
    confidence: float = 0.0
    handler: Optional[str] = None
    t_start: float = field(default_factory=time.monotonic)
