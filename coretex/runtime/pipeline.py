"""PipelineRunner — orchestrates the classifier → router → worker → executor pipeline.

This is the core of the v0.3.0 runtime. It retrieves components from the module
registry rather than importing them directly, satisfying Guardrail 2 (runtime
must never depend on modules).

v0.4.0 introduces PipelineStep, PipelineDefinition, and a PipelineRunner that
accepts a PipelineDefinition, enabling configurable pipelines. The default
pipeline preserves the original classifier → router → worker → executor behaviour.

Failure behaviour:
    ClassificationFailure   → fallback to intent=ambiguous, return clarification response
    WorkerFailure           → return worker failure response, set intent=ambiguous
    ToolExecutionFailure    → return worker failure response
    AgentParseFailure       → treat raw output as plain-text response (JSON parse error)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import httpx

from coretex.registry.module_registry import ModuleRegistry
from coretex.registry.tool_registry import ToolRegistry
from coretex.runtime.context import ExecutionContext
from coretex.runtime.executor import ToolExecutor, parse_agent_output

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline definition primitives
# ---------------------------------------------------------------------------

VALID_STEP_TYPES = frozenset({"classifier", "router", "worker", "tool_executor"})

DEFAULT_PIPELINE_NAME = "default"


@dataclass
class PipelineStep:
    """A single step in a pipeline definition.

    Attributes:
        component_type: The role of the component. Must be one of
            ``'classifier'``, ``'router'``, ``'worker'``, or ``'tool_executor'``.
        name: The name under which the component is registered in the
            ModuleRegistry (not used for ``'tool_executor'``).
    """

    component_type: Literal["classifier", "router", "worker", "tool_executor"]
    name: str

    def __post_init__(self) -> None:
        if self.component_type not in VALID_STEP_TYPES:
            raise ValueError(
                f"Invalid step component_type '{self.component_type}'. "
                f"Must be one of: {sorted(VALID_STEP_TYPES)}"
            )


@dataclass
class PipelineDefinition:
    """Describes an ordered sequence of pipeline steps to execute.

    Attributes:
        name: A unique identifier for this pipeline (used in observability logs).
        steps: Ordered list of PipelineStep objects describing the pipeline.
    """

    name: str
    steps: List[PipelineStep] = field(default_factory=list)

    def get_step(self, component_type: str) -> Optional[PipelineStep]:
        """Return the first step whose ``component_type`` matches, or ``None``."""
        for step in self.steps:
            if step.component_type == component_type:
                return step
        return None


def make_default_pipeline() -> PipelineDefinition:
    """Return the default CortX pipeline definition.

    This pipeline replicates the pre-v0.4.0 hardcoded behaviour:
    ClassifierBasic → RouterSimple → WorkerLLM → ToolExecutor.
    """
    return PipelineDefinition(
        name=DEFAULT_PIPELINE_NAME,
        steps=[
            PipelineStep(component_type="classifier", name="classifier_basic"),
            PipelineStep(component_type="router", name="router_simple"),
            PipelineStep(component_type="worker", name="worker_llm"),
            PipelineStep(component_type="tool_executor", name="tool_executor"),
        ],
    )


CLARIFY_RESPONSE = (
    "I'm not sure what you're asking. Could you provide more detail or clarify your request?"
)
_WORKER_FAILURE_RESPONSE = (
    "I'm sorry, I was unable to process your request right now. Please try again later."
)


class PipelineRunner:
    """Executes a CortX request pipeline defined by a PipelineDefinition.

    Components are resolved from the module registry at runtime, so the pipeline
    never hard-codes which classifier, router, or worker implementation is used.

    If no pipeline is supplied, the default pipeline is used, which preserves
    the pre-v0.4.0 behaviour:
        ClassifierBasic → RouterSimple → WorkerLLM → ToolExecutor.

    The pipeline follows these steps for every request:
        1. Classify  — determine intent and confidence via the registered classifier.
        2. Route     — select a handler deterministically based on intent.
        3. Execute   — invoke the worker (if routed to one), parse its JSON output,
                       and run any requested tool through ToolExecutor.

    All failure modes are handled gracefully; no request produces an unhandled
    exception that would result in a 500 response.
    """

    def __init__(
        self,
        module_registry: ModuleRegistry,
        tool_registry: ToolRegistry,
        pipeline: Optional[PipelineDefinition] = None,
    ) -> None:
        self._modules = module_registry
        self._executor = ToolExecutor(tool_registry)
        self._pipeline = pipeline if pipeline is not None else make_default_pipeline()

        # Resolve component names from pipeline steps (with sensible defaults).
        classifier_step = self._pipeline.get_step("classifier")
        router_step = self._pipeline.get_step("router")
        worker_step = self._pipeline.get_step("worker")

        self._classifier_name = classifier_step.name if classifier_step else "classifier_basic"
        self._router_name = router_step.name if router_step else "router_simple"
        self._worker_name = worker_step.name if worker_step else "worker_llm"

    async def run(self, context: ExecutionContext) -> Tuple[str, str, float]:
        """Execute the pipeline for the given *context*.

        Returns a tuple of ``(response_text, intent, confidence)``.
        """
        logger.info("event=request_received request_id=%s", context.request_id)
        logger.info(
            "event=pipeline_selected request_id=%s pipeline=%s",
            context.request_id,
            self._pipeline.name,
        )
        t_start = context.t_start

        # ------------------------------------------------------------------
        # Step 1: Classify
        # ------------------------------------------------------------------
        classifier = self._modules.get_classifier(self._classifier_name)

        logger.info("event=classifier_start request_id=%s classifier=%s", context.request_id, self._classifier_name)
        t_classify_start = time.monotonic()

        try:
            classification = await classifier.classify(context.user_input, context.request_id)
        except (httpx.HTTPError, httpx.RequestError) as exc:
            logger.error(
                "event=pipeline_classifier_failure request_id=%s error_type=%s error=%r",
                context.request_id,
                type(exc).__name__,
                str(exc),
            )
            classification = None

        t_classified = time.monotonic()
        classifier_latency_ms = int((t_classified - t_classify_start) * 1000)

        if classification is None:
            context.intent = "ambiguous"
            context.confidence = 0.0
        else:
            context.intent = classification.intent
            context.confidence = classification.confidence

        logger.info(
            "event=classifier_complete request_id=%s intent=%s confidence=%.2f duration_ms=%d",
            context.request_id,
            context.intent,
            context.confidence,
            classifier_latency_ms,
        )

        # ------------------------------------------------------------------
        # Step 2: Route
        # ------------------------------------------------------------------
        router = self._modules.get_router(self._router_name)
        handler = router.route(
            context.intent,
            request_id=context.request_id,
            user_input=context.user_input,
            confidence=context.confidence,
        )
        context.handler = handler

        logger.info(
            "event=router_selected request_id=%s intent=%s handler=%s",
            context.request_id,
            context.intent,
            handler,
        )

        # ------------------------------------------------------------------
        # Step 3: Execute handler
        # ------------------------------------------------------------------
        if handler == "clarify":
            response_text = CLARIFY_RESPONSE
            t_worker = t_classified
        else:
            logger.info(
                "event=worker_start request_id=%s worker=%s intent=%s",
                context.request_id,
                self._worker_name,
                context.intent,
            )
            t_worker_start = time.monotonic()

            try:
                worker = self._modules.get_worker(self._worker_name)
                response_text = await worker.generate(
                    context.user_input, context.intent, context.request_id
                )

                logger.info(
                    "event=worker_complete request_id=%s duration_ms=%d",
                    context.request_id,
                    int((time.monotonic() - t_worker_start) * 1000),
                )

                try:
                    action = parse_agent_output(response_text, request_id=context.request_id)
                    response_text = self._executor.execute(action, request_id=context.request_id)
                except json.JSONDecodeError:
                    # AgentParseFailure — LLM returned plain text instead of JSON.
                    logger.info(
                        "event=pipeline_agent_parse_failure request_id=%s "
                        "reason=json_decode_error treating as plain text",
                        context.request_id,
                    )
                except Exception as exc:
                    # ToolExecutionFailure — tool lookup or runtime exception.
                    logger.error(
                        "event=pipeline_tool_failure request_id=%s error_type=%s error=%r",
                        context.request_id,
                        type(exc).__name__,
                        str(exc),
                    )
                    response_text = _WORKER_FAILURE_RESPONSE

            except (httpx.HTTPError, httpx.RequestError) as exc:
                # WorkerFailure — Ollama unreachable or HTTP error.
                status = getattr(exc.response, "status_code", "N/A") if hasattr(exc, "response") else "N/A"
                body = ""
                if hasattr(exc, "response") and exc.response is not None:
                    try:
                        body = exc.response.text[:200]
                    except Exception:
                        pass
                logger.error(
                    "event=pipeline_worker_failure request_id=%s error_type=%s status=%s body=%r error=%r",
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
            "event=request_complete request_id=%s intent=%s confidence=%.2f handler=%s "
            "classifier_latency_ms=%d worker_latency_ms=%d total_latency_ms=%d",
            context.request_id,
            context.intent,
            context.confidence,
            context.handler,
            classifier_latency_ms,
            int((t_worker - t_classified) * 1000),
            total_latency_ms,
        )

        return response_text, context.intent, context.confidence
