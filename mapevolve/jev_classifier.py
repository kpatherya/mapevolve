"""Jev (TypeSafe) query classification adapter.

Jev is the primary classifier. The deterministic rule classifier is used only
when Jev cannot run or returns an invalid classification, and the result always
records which path produced it (``source``) and why a fallback happened.

The SDK (``langchain_typesafe``) is imported lazily so the stdlib core and CI
never need it; tests inject a fake decision model.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .contracts import QueryInterpretation, RoutingRequest
from .query_classifier import interpret_request

# Jev query types -> canonical MapEvolve families. landmark_identification has
# no MVP family and therefore maps to "unknown" (abstains).
QUERY_TYPE_TO_FAMILY = {
    "cross_view_matching": "same_building_verification",
    "temporal_comparison": "change_captioning",
    "factual_retrieval": "evidence_backed_investigation",
    "spatial_reasoning": "proximity_computation",
    "landmark_identification": "unknown",
}
DIFFICULTIES = ("easy", "moderate", "hard")

# Fields that must never reach the classifier (answer keys / labels).
FORBIDDEN_STATE_KEYS = frozenset(
    {
        "ground_truth",
        "ground_truth_raw",
        "rubric",
        "reference",
        "reference_answer",
        "expected_decision",
        "expected_same_building",
        "answerable",
        "difficulty",
        "case_id",
        "building",
    }
)

_QUERY_TYPE_CRITERIA = {
    "landmark_identification": "Name or identify a building or landmark shown in an image.",
    "cross_view_matching": "Decide whether two images show the same building or place.",
    "temporal_comparison": "Compare images of the same place at different times; describe changes.",
    "spatial_reasoning": "Reason about locations, distances, directions, or layout between places.",
    "factual_retrieval": "Ask for facts not visible in pixels, such as history, dates, names, or records.",
}
_DIFFICULTY_CRITERIA = [
    "easy: the answer is directly visible.",
    "moderate: viewpoint, season, or lighting differences must be separated from real change.",
    "hard: large time gaps, archival photos, strong viewpoint change, or evidence beyond the images.",
]


class DecisionModel(Protocol):
    def invoke(self, input: dict[str, Any], config: Any = None) -> Any: ...


class JevOutputError(ValueError):
    """Jev returned a classification that cannot be mapped to the taxonomy."""


@dataclass(frozen=True)
class ClassificationResult:
    interpretation: QueryInterpretation
    source: str  # "jev" | "rules" | "rules_fallback"
    difficulty: str | None = None
    difficulty_probabilities: Mapping[str, float] = field(default_factory=dict)
    query_type: str | None = None
    query_type_probabilities: Mapping[str, float] = field(default_factory=dict)
    fallback_reason: str | None = None
    jev_model: str | None = None
    jev_request_id: str | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "difficulty": self.difficulty,
            "difficulty_probabilities": dict(self.difficulty_probabilities),
            "query_type": self.query_type,
            "query_type_probabilities": dict(self.query_type_probabilities),
            "fallback_reason": self.fallback_reason,
            "jev_model": self.jev_model,
            "jev_request_id": self.jev_request_id,
            "latency_ms": self.latency_ms,
        }


def rules_classification(request: RoutingRequest) -> ClassificationResult:
    return ClassificationResult(interpretation=interpret_request(request), source="rules")


def build_state(request: RoutingRequest, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Jev input: query text, image count, and non-label metadata only (never image paths)."""
    metadata = dict(metadata or {})
    leaked = FORBIDDEN_STATE_KEYS & set(metadata)
    if leaked:
        raise ValueError(f"Refusing to send answer-bearing metadata to Jev: {sorted(leaked)}")
    state: dict[str, Any] = {"question": request.query, "num_images": len(request.images)}
    if metadata:
        state["query_metadata"] = metadata
    return state


def build_questions() -> dict[str, Any]:
    from langchain_typesafe import Choice, Score  # lazy: optional dependency

    return {
        "query_type": Choice(
            instructions="What kind of visual question is the user asking?",
            criteria=_QUERY_TYPE_CRITERIA,
        ),
        "difficulty": Score(
            instructions="How difficult is this query for a vision-language model to answer correctly?",
            criteria=_DIFFICULTY_CRITERIA,
        ),
    }


def _parse(response: Any, request: RoutingRequest) -> ClassificationResult:
    try:
        choice = response.choices["query_type"]
        label = choice.choice
        type_probs = {str(k): float(v) for k, v in choice.probabilities.items()}
        confidence = float(choice.confidence)
        score = response.scores["difficulty"]
        diff_probs = {DIFFICULTIES[int(i)]: float(p) for i, p in score.probabilities.items()}
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
        raise JevOutputError(f"malformed Jev response: {error}") from error
    if label not in QUERY_TYPE_TO_FAMILY:
        raise JevOutputError(f"unknown Jev query_type {label!r}")
    if not 0.0 <= confidence <= 1.0:
        raise JevOutputError("Jev confidence out of range")
    if not diff_probs:
        raise JevOutputError("Jev returned no difficulty distribution")
    difficulty = max(diff_probs, key=diff_probs.get)
    interpretation = QueryInterpretation(
        request_id=request.request_id,
        family=QUERY_TYPE_TO_FAMILY[label],
        rule_id=f"jev.query_type.{label}",
        confidence=confidence,
    )
    return ClassificationResult(
        interpretation=interpretation,
        source="jev",
        difficulty=difficulty,
        difficulty_probabilities=diff_probs,
        query_type=label,
        query_type_probabilities=type_probs,
        jev_model=getattr(response, "model", None),
        jev_request_id=getattr(response, "request_id", None),
    )


class JevQueryClassifier:
    """Callable ``request -> ClassificationResult``; falls back to rules on failure."""

    def __init__(
        self,
        decision_model: DecisionModel | None = None,
        *,
        model: str = "jev-latest",
        timeout: float = 20.0,
        questions_factory: Callable[[], dict[str, Any]] = build_questions,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._decision_model = decision_model
        self._model_name = model
        self._timeout = timeout
        self._questions_factory = questions_factory
        self._metadata = metadata

    def _get_model(self) -> DecisionModel:
        if self._decision_model is not None:
            return self._decision_model
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        from langchain_typesafe import TypeSafeClassifier  # lazy: optional dependency

        self._decision_model = TypeSafeClassifier(model=self._model_name, timeout=self._timeout)
        return self._decision_model

    def __call__(self, request: RoutingRequest) -> ClassificationResult:
        state = build_state(request, self._metadata)  # leak guard errors are not swallowed
        started = time.perf_counter()
        try:
            model = self._get_model()
            response = model.invoke({"state": state, "questions": self._questions_factory()})
            result = _parse(response, request)
            latency = (time.perf_counter() - started) * 1000.0
            return ClassificationResult(**{**result.__dict__, "latency_ms": latency})
        except Exception as error:  # Jev unavailable or invalid output: explicit, labelled fallback
            fallback = interpret_request(request)
            return ClassificationResult(
                interpretation=fallback,
                source="rules_fallback",
                fallback_reason=f"{type(error).__name__}: {error}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
