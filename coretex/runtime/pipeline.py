"""PipelineRunner — orchestrates the classifier → router → worker → executor pipeline.

This is the core of the v0.3.0 runtime. It retrieves components from the module
registry rather than importing them directly, satisfying Guardrail 2 (runtime
must never depend on modules).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Tuple

import httpx

from coretex.registry.module_registry import ModuleRegistry
from coretex.registry.tool_registry import ToolRegistry
from coretex.runtime.context import ExecutionContext
from coretex.runtime.executor import ToolExecutor, parse_agent_output

logger = logging.getLogger(__name__)

_CLARIFY_RESPONSE = (
    "I'm not sure what you're asking. Could you provide more detail or clarify your request?"
)
_WORKER_FAILURE_RESPONSE = (
    "I'm sorry, I was unable to process your request right now. Please try again later."
)


class PipelineRunner:
    """Executes the standard CortX request pipeline using registered modules.

    Components are resolved from the module registry at runtime, so the pipeline
    never hard-codes which classifier, router, or worker implementation is used.
    """

    def __init__(
        self,
        module_registry: ModuleRegistry,
        tool_registry: ToolRegistry,
        classifier_name: str = "classifier_basic",
        router_name: str = "router_simple",
        worker_name: str = "worker_llm",
    ) -> None:
        self._modules = module_registry
        self._executor = ToolExecutor(tool_registry)
        self._classifier_name = classifier_name
        self._router_name = router_name
        self._worker_name = worker_name

    async def run(self, context: ExecutionContext) -> Tuple[str, str, float]:
        """Execute the pipeline for the given *context*.

        Returns ``(response_text, intent, confidence)``.
        """
        logger.info("event=request_received request_id=%s", context.request_id)
        t_start = context.t_start

        # ------------------------------------------------------------------
        # Step 1: Classify
        # ------------------------------------------------------------------
        classifier = self._modules.get_classifier(self._classifier_name)
        classification = await classifier.classify(context.user_input, context.request_id)
        t_classified = time.monotonic()

        context.intent = classification.intent
        context.confidence = classification.confidence

        # ------------------------------------------------------------------
        # Step 2: Route
        # ------------------------------------------------------------------
        router = self._modules.get_router(self._router_name)
        handler = router.route(
            classification.intent,
            request_id=context.request_id,
            user_input=context.user_input,
            confidence=classification.confidence,
        )
        context.handler = handler

        # ------------------------------------------------------------------
        # Step 3: Execute handler
        # ------------------------------------------------------------------
        if handler == "clarify":
            response_text = _CLARIFY_RESPONSE
            t_worker = t_classified
        else:
            logger.info(
                "event=agent_selected request_id=%s agent=worker",
                context.request_id,
            )
            try:
                worker = self._modules.get_worker(self._worker_name)
                response_text = await worker.generate(
                    context.user_input, classification.intent, context.request_id
                )
                try:
                    action = parse_agent_output(response_text, request_id=context.request_id)
                    response_text = self._executor.execute(action, request_id=context.request_id)
                except json.JSONDecodeError:
                    # LLM returned plain text instead of JSON — treat as direct reply.
                    pass
                except Exception as exc:
                    logger.error(
                        "event=tool_execution_error request_id=%s error_type=%s error=%r",
                        context.request_id,
                        type(exc).__name__,
                        str(exc),
                    )
                    response_text = _WORKER_FAILURE_RESPONSE

            except (httpx.HTTPError, httpx.RequestError) as exc:
                status = getattr(exc.response, "status_code", "N/A") if hasattr(exc, "response") else "N/A"
                body = ""
                if hasattr(exc, "response") and exc.response is not None:
                    try:
                        body = exc.response.text[:200]
                    except Exception:
                        pass
                logger.error(
                    "event=worker_error request_id=%s error_type=%s status=%s body=%r error=%r",
                    context.request_id,
                    type(exc).__name__,
                    status,
                    body,
                    str(exc) or repr(exc),
                )
                response_text = _WORKER_FAILURE_RESPONSE
                context.intent = "ambiguous"
                context.confidence = 0.0

            t_worker = time.monotonic()

        total_latency_ms = int((time.monotonic() - t_start) * 1000)
        logger.info(
            "event=request_complete request_id=%s intent=%s confidence=%.2f "
            "classifier_latency_ms=%d worker_latency_ms=%d total_latency_ms=%d",
            context.request_id,
            context.intent,
            context.confidence,
            int((t_classified - t_start) * 1000),
            int((t_worker - t_classified) * 1000),
            total_latency_ms,
        )

        return response_text, context.intent, context.confidence
