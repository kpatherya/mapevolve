"""End-to-end routing pipeline for the MapEvolve MVP vertical slice."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .backends import Backend
from .contracts import BackendResult, QueryInterpretation, RoutingDecision, RoutingRequest
from .pareto_policy import (
    DEFAULT_ACCURACY_MARGIN,
    DEFAULT_PROFILES_PATH,
    PolicySelection,
    decide_with_pareto_policy,
)
from .jev_classifier import ClassificationResult, rules_classification


class PipelineConfigurationError(ValueError):
    """Raised when the pipeline is missing required configuration."""


PayloadBuilder = Callable[[RoutingRequest], dict[str, Any]]
Classifier = Callable[[RoutingRequest], ClassificationResult]
PolicySelector = Callable[[RoutingRequest, QueryInterpretation], PolicySelection]


@dataclass(frozen=True)
class PipelineTrace:
    """Structured trace for one end-to-end pipeline execution."""

    request: RoutingRequest
    interpretation: QueryInterpretation
    decision: RoutingDecision
    backend_result: BackendResult | None
    policy_selection: PolicySelection
    dry_run: bool
    dry_run_payload: dict[str, Any] | None
    latency_ms: float
    estimated_cost_usd: float | None
    classification: ClassificationResult | None = None
    dry_run_payload_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "interpretation": self.interpretation.to_dict(),
            "decision": self.decision.to_dict(),
            "backend_result": None if self.backend_result is None else self.backend_result.to_dict(),
            "policy": {
                "profile_version": self.policy_selection.profile_version,
                "selected_model": None
                if self.policy_selection.selected_model is None
                else self.policy_selection.selected_model.model_id,
                "accuracy_margin": self.policy_selection.accuracy_margin,
                "pareto_frontier": [model.model_id for model in self.policy_selection.pareto_frontier],
                "details": self.policy_selection.policy_details,
            },
            "classification": None if self.classification is None else self.classification.to_dict(),
            "dry_run": self.dry_run,
            "dry_run_payload": self.dry_run_payload,
            "dry_run_payload_bytes": self.dry_run_payload_bytes,
            "latency_ms": self.latency_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"^data:([^;]+);base64,.*$", r"data:\1;base64,<redacted>", value)
    return value


def _resolve_builder(
    backend_id: str,
    backend_registry: Mapping[str, Backend] | None,
    payload_builders: Mapping[str, PayloadBuilder] | None,
) -> PayloadBuilder:
    if payload_builders is not None and backend_id in payload_builders:
        return payload_builders[backend_id]
    if backend_registry is not None and backend_id in backend_registry:
        backend = backend_registry[backend_id]
        builder = getattr(backend, "build_payload", None)
        if callable(builder):
            return builder
    raise PipelineConfigurationError(
        f"no payload builder configured for backend_id={backend_id!r}; dry-run requires payload construction"
    )


def run_pipeline(
    request: RoutingRequest,
    *,
    backend_registry: Mapping[str, Backend] | None = None,
    payload_builders: Mapping[str, PayloadBuilder] | None = None,
    dry_run: bool = False,
    accuracy_margin: float = DEFAULT_ACCURACY_MARGIN,
    profiles_path=DEFAULT_PROFILES_PATH,
    classifier: Classifier | None = None,
    policy_selector: PolicySelector | None = None,
) -> PipelineTrace:
    """Run validate -> interpret -> decide -> execute -> trace."""
    started = time.perf_counter()

    classification = (classifier or rules_classification)(request)
    interpretation = classification.interpretation
    if policy_selector is not None:
        policy_selection = policy_selector(request, interpretation)
    else:
        policy_selection = decide_with_pareto_policy(
            request,
            interpretation,
            accuracy_margin=accuracy_margin,
            profiles_path=profiles_path,
        )
    decision = policy_selection.decision

    backend_result: BackendResult | None = None
    dry_run_payload: dict[str, Any] | None = None
    dry_run_payload_bytes: int | None = None
    estimated_cost_usd: float | None = None

    if decision.action == "execute":
        backend_id = decision.backend_id
        if backend_id is None:
            raise PipelineConfigurationError("execute decision is missing backend_id")

        if dry_run:
            builder = _resolve_builder(backend_id, backend_registry, payload_builders)
            raw_payload = builder(request)
            dry_run_payload_bytes = len(json.dumps(raw_payload).encode("utf-8"))
            dry_run_payload = _sanitize_payload(raw_payload)
            details = policy_selection.policy_details
            if details is not None:
                estimated_cost_usd = details["estimated_cost_usd"]
            elif policy_selection.selected_model is not None:
                estimated_cost_usd = float(len(request.images)) * policy_selection.selected_model.image_cost_usd
        else:
            if backend_registry is None or backend_id not in backend_registry:
                raise PipelineConfigurationError(
                    f"no backend configured for execute decision backend_id={backend_id!r}"
                )
            backend = backend_registry[backend_id]
            # Execute exactly the named backend: no fallback if execution fails.
            backend_result = backend.execute(request)
            estimated_cost_usd = backend_result.estimated_cost_usd

    latency_ms = (time.perf_counter() - started) * 1000.0
    return PipelineTrace(
        request=request,
        interpretation=interpretation,
        decision=decision,
        backend_result=backend_result,
        policy_selection=policy_selection,
        dry_run=dry_run,
        dry_run_payload=dry_run_payload,
        latency_ms=latency_ms,
        estimated_cost_usd=estimated_cost_usd,
        classification=classification,
        dry_run_payload_bytes=dry_run_payload_bytes,
    )
