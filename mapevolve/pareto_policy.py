"""Static profile loading and Pareto-based policy selection for routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import FAMILIES, QueryInterpretation, RoutingDecision, RoutingRequest
from .query_classifier import non_execution_decision

PROFILES_SCHEMA_VERSION = 1
DEFAULT_ACCURACY_MARGIN = 0.02
DEFAULT_PROFILES_PATH = Path(__file__).with_name("model_profiles.json")


class ProfileValidationError(ValueError):
    """Raised when static model profiles are malformed or unsupported."""


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileValidationError(f"{field_name} must be a number")
    return float(value)


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    offline_accuracy: float
    input_cost_usd_per_1m_tokens: float
    image_cost_usd: float

    def __post_init__(self) -> None:
        _require_non_empty_string(self.model_id, "model_id")
        if not 0.0 <= self.offline_accuracy <= 1.0:
            raise ProfileValidationError("offline_accuracy must be in [0, 1]")
        if self.input_cost_usd_per_1m_tokens < 0.0:
            raise ProfileValidationError("input_cost_usd_per_1m_tokens must be >= 0")
        if self.image_cost_usd < 0.0:
            raise ProfileValidationError("image_cost_usd must be >= 0")

    @property
    def total_cost_proxy(self) -> float:
        # Proxy metric for policy ranking: one image call plus one token-cost bucket.
        return self.input_cost_usd_per_1m_tokens + self.image_cost_usd


@dataclass(frozen=True)
class FamilyProfiles:
    recommended_model: str | None
    models: tuple[ModelProfile, ...]

    def __post_init__(self) -> None:
        if self.recommended_model is not None:
            _require_non_empty_string(self.recommended_model, "recommended_model")
            model_ids = {model.model_id for model in self.models}
            if self.recommended_model not in model_ids:
                raise ProfileValidationError(
                    "recommended_model must reference one of the family's model_id values"
                )


@dataclass(frozen=True)
class ProfilesCatalog:
    profile_version: str
    accuracy_margin: float
    families: dict[str, FamilyProfiles]
    schema_version: int = PROFILES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROFILES_SCHEMA_VERSION:
            raise ProfileValidationError(
                f"schema_version must be {PROFILES_SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        _require_non_empty_string(self.profile_version, "profile_version")
        if self.accuracy_margin < 0.0:
            raise ProfileValidationError("accuracy_margin must be >= 0")
        expected = set(FAMILIES)
        found = set(self.families.keys())
        missing = expected - found
        if missing:
            raise ProfileValidationError(f"profiles missing family entries: {sorted(missing)}")
        unexpected = found - expected
        if unexpected:
            raise ProfileValidationError(f"profiles include unsupported families: {sorted(unexpected)}")

        # MVP requires static candidates for same-building and empty stubs elsewhere.
        same_building = self.families["same_building_verification"]
        if len(same_building.models) < 2:
            raise ProfileValidationError(
                "same_building_verification must include at least two model profiles"
            )
        for family in FAMILIES:
            if family == "same_building_verification":
                continue
            if self.families[family].models:
                raise ProfileValidationError(
                    f"{family} must remain an empty/stub profile for this MVP stage"
                )

    def for_family(self, family: str) -> FamilyProfiles:
        if family not in self.families:
            raise ProfileValidationError(f"unsupported family: {family!r}")
        return self.families[family]


@dataclass(frozen=True)
class PolicySelection:
    decision: RoutingDecision
    selected_model: ModelProfile | None
    profile_version: str | None
    pareto_frontier: tuple[ModelProfile, ...]
    accuracy_margin: float


def _parse_model(payload: dict[str, Any], index: int) -> ModelProfile:
    if not isinstance(payload, dict):
        raise ProfileValidationError(f"models[{index}] must be an object")
    required = {
        "model_id",
        "offline_accuracy",
        "input_cost_usd_per_1m_tokens",
        "image_cost_usd",
    }
    keys = set(payload.keys())
    unknown = keys - required
    if unknown:
        raise ProfileValidationError(f"models[{index}] has unknown fields: {sorted(unknown)}")
    missing = required - keys
    if missing:
        raise ProfileValidationError(f"models[{index}] is missing fields: {sorted(missing)}")
    return ModelProfile(
        model_id=_require_non_empty_string(payload["model_id"], f"models[{index}].model_id"),
        offline_accuracy=_require_number(payload["offline_accuracy"], f"models[{index}].offline_accuracy"),
        input_cost_usd_per_1m_tokens=_require_number(
            payload["input_cost_usd_per_1m_tokens"],
            f"models[{index}].input_cost_usd_per_1m_tokens",
        ),
        image_cost_usd=_require_number(payload["image_cost_usd"], f"models[{index}].image_cost_usd"),
    )


def load_profiles(path: Path = DEFAULT_PROFILES_PATH) -> ProfilesCatalog:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ProfileValidationError(f"profile file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ProfileValidationError(f"profile file JSON decode failed: {error}") from error

    if not isinstance(payload, dict):
        raise ProfileValidationError("profile file must contain a JSON object")

    required_root = {"schema_version", "profile_version", "accuracy_margin", "families"}
    root_keys = set(payload.keys())
    unknown_root = root_keys - required_root
    if unknown_root:
        raise ProfileValidationError(f"profile file has unknown root fields: {sorted(unknown_root)}")
    missing_root = required_root - root_keys
    if missing_root:
        raise ProfileValidationError(f"profile file is missing root fields: {sorted(missing_root)}")

    families_payload = payload["families"]
    if not isinstance(families_payload, dict):
        raise ProfileValidationError("families must be an object")

    families: dict[str, FamilyProfiles] = {}
    for family, family_payload in families_payload.items():
        if not isinstance(family_payload, dict):
            raise ProfileValidationError(f"families.{family} must be an object")
        expected_family_keys = {"recommended_model", "models"}
        family_keys = set(family_payload.keys())
        extra = family_keys - expected_family_keys
        if extra:
            raise ProfileValidationError(f"families.{family} has unknown fields: {sorted(extra)}")
        missing = expected_family_keys - family_keys
        if missing:
            raise ProfileValidationError(f"families.{family} is missing fields: {sorted(missing)}")

        models_payload = family_payload["models"]
        if not isinstance(models_payload, list):
            raise ProfileValidationError(f"families.{family}.models must be a list")
        models = tuple(_parse_model(model_payload, index) for index, model_payload in enumerate(models_payload))
        families[family] = FamilyProfiles(
            recommended_model=family_payload["recommended_model"],
            models=models,
        )

    return ProfilesCatalog(
        schema_version=payload["schema_version"],
        profile_version=payload["profile_version"],
        accuracy_margin=_require_number(payload["accuracy_margin"], "accuracy_margin"),
        families=families,
    )


def pareto_frontier(models: tuple[ModelProfile, ...]) -> tuple[ModelProfile, ...]:
    frontier: list[ModelProfile] = []
    for candidate in models:
        dominated = False
        for other in models:
            if other is candidate:
                continue
            no_worse_cost = other.total_cost_proxy <= candidate.total_cost_proxy
            no_worse_accuracy = other.offline_accuracy >= candidate.offline_accuracy
            strictly_better = (
                other.total_cost_proxy < candidate.total_cost_proxy
                or other.offline_accuracy > candidate.offline_accuracy
            )
            if no_worse_cost and no_worse_accuracy and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda model: (model.total_cost_proxy, model.model_id)))


def choose_model(frontier: tuple[ModelProfile, ...], accuracy_margin: float) -> ModelProfile:
    if not frontier:
        raise ProfileValidationError("no eligible model profiles available")
    if accuracy_margin < 0.0:
        raise ProfileValidationError("accuracy_margin must be >= 0")

    best_accuracy = max(model.offline_accuracy for model in frontier)
    threshold = best_accuracy - accuracy_margin
    eligible = [model for model in frontier if model.offline_accuracy >= threshold]
    if not eligible:
        raise ProfileValidationError("no model meets the configured accuracy margin")
    return min(eligible, key=lambda model: (model.total_cost_proxy, model.model_id))


def decide_with_pareto_policy(
    request: RoutingRequest,
    interpretation: QueryInterpretation,
    *,
    accuracy_margin: float = DEFAULT_ACCURACY_MARGIN,
    profiles_path: Path = DEFAULT_PROFILES_PATH,
) -> PolicySelection:
    non_exec = non_execution_decision(request, interpretation)
    if non_exec is not None:
        return PolicySelection(
            decision=non_exec,
            selected_model=None,
            profile_version=None,
            pareto_frontier=(),
            accuracy_margin=accuracy_margin,
        )

    catalog = load_profiles(profiles_path)
    family_profiles = catalog.for_family(interpretation.family)
    if not family_profiles.models:
        raise ProfileValidationError(
            f"family {interpretation.family!r} has no executable model profiles configured"
        )

    frontier = pareto_frontier(family_profiles.models)
    selected = choose_model(frontier, accuracy_margin=accuracy_margin)
    decision = RoutingDecision(
        request_id=request.request_id,
        family=interpretation.family,
        action="execute",
        reason_code="ready",
        backend_id=selected.model_id,
    )
    return PolicySelection(
        decision=decision,
        selected_model=selected,
        profile_version=catalog.profile_version,
        pareto_frontier=frontier,
        accuracy_margin=accuracy_margin,
    )
