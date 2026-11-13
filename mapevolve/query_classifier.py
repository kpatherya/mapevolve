"""Deterministic rule-based query classification for MapEvolve MVP."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import QueryInterpretation, RoutingDecision, RoutingRequest


@dataclass(frozen=True)
class _Rule:
    family: str
    rule_id: str
    patterns: tuple[re.Pattern[str], ...]


_RULES: tuple[_Rule, ...] = (
    _Rule(
        family="same_building_verification",
        rule_id="rule.same_building.keyword_v1",
        patterns=(
            re.compile(r"\bsame building\b"),
            re.compile(r"\b(?:are|is)\s+(?:these|this)\s+the\s+same\b"),
            re.compile(r"\b(?:same|identical)\s+(?:place|location|structure)\b"),
            re.compile(r"\bdo(?:es)?\s+(?:these|this)\s+show\s+the\s+same\b"),
        ),
    ),
    _Rule(
        family="change_captioning",
        rule_id="rule.change_captioning.keyword_v1",
        patterns=(
            re.compile(r"\bsimilarities?\s+and\s+differences?\b"),
            re.compile(r"\bwhat\s+changed\b"),
            re.compile(r"\bcompare\b.*\bimages?\b"),
            re.compile(r"\bdifferences?\s+between\b"),
        ),
    ),
    _Rule(
        family="evidence_backed_investigation",
        rule_id="rule.evidence_backed_investigation.keyword_v1",
        patterns=(
            re.compile(r"\bwho\s+founded\b"),
            re.compile(r"\bwho\s+built\b"),
            re.compile(r"\bwhen\s+was\b"),
            re.compile(r"\bwhat\s+year\b"),
            re.compile(r"\binvestigate\b"),
        ),
    ),
    _Rule(
        family="proximity_computation",
        rule_id="rule.proximity_computation.keyword_v1",
        patterns=(
            re.compile(r"\bhow\s+close\b"),
            re.compile(r"\bdistance\s+between\b"),
            re.compile(r"\bhow\s+far\b"),
            re.compile(r"\bwithin\s+\d+(?:\.\d+)?\s*(?:m|km|meters?|kilometers?)\b"),
            re.compile(r"\bnear(?:est)?\b"),
        ),
    ),
)


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def interpret_request(request: RoutingRequest) -> QueryInterpretation:
    """Classify a request into one of the fixed taxonomy families."""
    normalized_query = _normalize_query(request.query)
    for rule in _RULES:
        for pattern in rule.patterns:
            if pattern.search(normalized_query):
                return QueryInterpretation(
                    request_id=request.request_id,
                    family=rule.family,
                    rule_id=rule.rule_id,
                    confidence=1.0,
                )
    return QueryInterpretation(
        request_id=request.request_id,
        family="unknown",
        rule_id="rule.unknown.fallback_v1",
        confidence=0.0,
    )


def non_execution_decision(
    request: RoutingRequest, interpretation: QueryInterpretation
) -> RoutingDecision | None:
    """Return a decision only for non-execution paths.

    Returns:
      - `request_evidence` when same-building verification does not have exactly two images.
      - `abstain` for unsupported known families and unknown queries.
      - `None` for executable same-building requests with exactly two images.
    """
    if interpretation.family == "same_building_verification":
        if len(request.images) != 2:
            return RoutingDecision(
                request_id=request.request_id,
                family=interpretation.family,
                action="request_evidence",
                reason_code="missing_required_images",
                required_images=2,
            )
        return None

    if interpretation.family == "unknown":
        return RoutingDecision(
            request_id=request.request_id,
            family=interpretation.family,
            action="abstain",
            reason_code="unknown_query_family",
        )

    return RoutingDecision(
        request_id=request.request_id,
        family=interpretation.family,
        action="abstain",
        reason_code="capability_not_yet_supported",
    )


def classify_and_decide(request: RoutingRequest) -> tuple[QueryInterpretation, RoutingDecision | None]:
    """Convenience helper for pipeline stages before policy/backend selection."""
    interpretation = interpret_request(request)
    return interpretation, non_execution_decision(request, interpretation)
