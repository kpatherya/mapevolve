"""Accuracy/cost tradeoff selection over the Pareto frontier (profile schema v2).

Modes (exactly one, explicit):

* ``utility``: maximise ``accuracy - lam * cost / max_cost`` over the frontier;
* ``budget``: highest accuracy whose estimated request cost <= ``max_cost_usd``;
* ``target``: cheapest model whose accuracy >= ``min_accuracy``.

``budget`` and ``target`` raise ``NoEligibleModelError`` when nothing qualifies;
the policy never silently picks a worse model. Accuracy is the Wilson lower
bound by default (``use_lower_bound=True``), the conservative family estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import QueryInterpretation, RoutingDecision, RoutingRequest
from .pareto_policy import ModelProfile, PolicySelection, ProfileValidationError
from .profile_builder import EvaluatedProfile, FamilyAggregate, ModelSpec
from .query_classifier import non_execution_decision

MODES = ("utility", "budget", "target")


class NoEligibleModelError(ProfileValidationError):
    """No frontier model satisfies the configured tradeoff constraint."""


@dataclass(frozen=True)
class CostAssumptions:
    prompt_overhead_tokens: int = 300
    chars_per_token: float = 4.0
    output_tokens: int = 100


@dataclass(frozen=True)
class TradeoffConfig:
    mode: str = "utility"
    lam: float = 0.1
    max_cost_usd: float | None = None
    min_accuracy: float | None = None
    use_lower_bound: bool = True

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ProfileValidationError(f"mode must be one of {MODES}")
        if self.mode == "utility" and self.lam < 0:
            raise ProfileValidationError("lam must be >= 0")
        if self.mode == "budget" and (self.max_cost_usd is None or self.max_cost_usd < 0):
            raise ProfileValidationError("budget mode requires max_cost_usd >= 0")
        if self.mode == "target" and (self.min_accuracy is None or not 0 <= self.min_accuracy <= 1):
            raise ProfileValidationError("target mode requires min_accuracy in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "lam": self.lam,
            "max_cost_usd": self.max_cost_usd,
            "min_accuracy": self.min_accuracy,
            "use_lower_bound": self.use_lower_bound,
        }


@dataclass(frozen=True)
class Candidate:
    model_id: str
    accuracy: float
    cost_usd: float
    aggregate: FamilyAggregate


def estimate_request_cost(
    model: ModelSpec, request: RoutingRequest, assumptions: CostAssumptions = CostAssumptions()
) -> float:
    text_tokens = assumptions.prompt_overhead_tokens + len(request.query) / assumptions.chars_per_token
    input_tokens = text_tokens + model.tokens_per_image * len(request.images)
    return (
        input_tokens * model.input_cost_usd_per_1m_tokens
        + assumptions.output_tokens * model.output_cost_usd_per_1m_tokens
    ) / 1_000_000.0


def pareto_candidates(candidates: list[Candidate]) -> list[Candidate]:
    frontier = []
    for c in candidates:
        dominated = any(
            o is not c
            and o.cost_usd <= c.cost_usd
            and o.accuracy >= c.accuracy
            and (o.cost_usd < c.cost_usd or o.accuracy > c.accuracy)
            for o in candidates
        )
        if not dominated:
            frontier.append(c)
    return sorted(frontier, key=lambda c: (c.cost_usd, c.model_id))


def select_candidate(frontier: list[Candidate], config: TradeoffConfig) -> Candidate:
    if not frontier:
        raise NoEligibleModelError("no eligible model profiles available")
    if config.mode == "target":
        ok = [c for c in frontier if c.accuracy >= config.min_accuracy]
        if not ok:
            raise NoEligibleModelError(f"no model reaches min_accuracy={config.min_accuracy}")
        return min(ok, key=lambda c: (c.cost_usd, c.model_id))
    if config.mode == "budget":
        ok = [c for c in frontier if c.cost_usd <= config.max_cost_usd]
        if not ok:
            raise NoEligibleModelError(f"no model fits max_cost_usd={config.max_cost_usd}")
        return max(ok, key=lambda c: (c.accuracy, -c.cost_usd, c.model_id))
    max_cost = max(c.cost_usd for c in frontier) or 1.0
    return max(frontier, key=lambda c: (c.accuracy - config.lam * c.cost_usd / max_cost, -c.cost_usd, c.model_id))


def decide_with_tradeoff_policy(
    request: RoutingRequest,
    interpretation: QueryInterpretation,
    profile: EvaluatedProfile,
    config: TradeoffConfig,
    assumptions: CostAssumptions = CostAssumptions(),
) -> PolicySelection:
    non_exec = non_execution_decision(request, interpretation)
    if non_exec is not None:
        return PolicySelection(non_exec, None, None, (), None, None)

    aggregates, excluded = profile.family_aggregates(interpretation.family)
    if not aggregates:
        raise ProfileValidationError(
            f"family {interpretation.family!r} has no evaluated model profiles configured"
        )
    candidates = [
        Candidate(
            a.model_id,
            a.accuracy_lower_bound if config.use_lower_bound else a.accuracy,
            estimate_request_cost(profile.models[a.model_id], request, assumptions),
            a,
        )
        for a in aggregates
    ]
    frontier = pareto_candidates(candidates)
    chosen = select_candidate(frontier, config)
    spec = profile.models[chosen.model_id]

    def _as_profile(c: Candidate) -> ModelProfile:
        s = profile.models[c.model_id]
        return ModelProfile(
            model_id=c.model_id,
            offline_accuracy=c.accuracy,
            input_cost_usd_per_1m_tokens=s.input_cost_usd_per_1m_tokens,
            image_cost_usd=s.tokens_per_image * s.input_cost_usd_per_1m_tokens / 1_000_000.0,
        )

    decision = RoutingDecision(
        request_id=request.request_id,
        family=interpretation.family,
        action="execute",
        reason_code="ready",
        backend_id=chosen.model_id,
    )
    details = {
        "policy": "tradeoff",
        "tradeoff": config.to_dict(),
        "cost_assumptions": {
            "prompt_overhead_tokens": assumptions.prompt_overhead_tokens,
            "chars_per_token": assumptions.chars_per_token,
            "output_tokens": assumptions.output_tokens,
        },
        "estimated_cost_usd": chosen.cost_usd,
        "selected_accuracy": chosen.accuracy,
        "selected_n": chosen.aggregate.n,
        "benchmarks": list(chosen.aggregate.benchmarks),
        "excluded_incomplete_coverage": list(excluded),
        "frontier": [
            {"model_id": c.model_id, "accuracy": c.accuracy, "cost_usd": c.cost_usd, "n": c.aggregate.n}
            for c in frontier
        ],
        "selected_tier": spec.tier,
    }
    return PolicySelection(
        decision=decision,
        selected_model=_as_profile(chosen),
        profile_version=profile.profile_version,
        pareto_frontier=tuple(_as_profile(c) for c in frontier),
        accuracy_margin=None,
        policy_details=details,
    )
