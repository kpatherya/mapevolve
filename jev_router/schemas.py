"""Plain data types shared across classification, routing, and execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

QUERY_TYPES = (
    "landmark_identification",
    "cross_view_matching",
    "temporal_comparison",
    "spatial_reasoning",
    "factual_retrieval",
)
DIFFICULTIES = ("easy", "moderate", "hard")
VISUAL_EXPERTISE = (
    "general_scene_understanding",
    "architectural_detail",
    "historical_photo_interpretation",
    "fine_grained_change_detection",
    "spatial_geometry",
)
MODES = ("economy", "balanced", "accuracy")

# The benchmark labels its middle tier "medium"; the router uses "moderate".
DIFFICULTY_ALIASES = {"medium": "moderate", "simple": "easy", "hardest": "hard"}


def normalize_difficulty(value: str) -> str:
    value = value.strip().lower()
    return DIFFICULTY_ALIASES.get(value, value)


@dataclass
class LabelDistribution:
    """A categorical decision with its full probability distribution."""

    label: str
    probabilities: dict[str, float]
    confidence: Optional[float] = None  # Jev: concentration of the distribution


@dataclass
class QueryClassification:
    query_type: LabelDistribution
    difficulty: LabelDistribution
    visual_expertise: LabelDistribution
    needs_image_retrieval: float      # probability in [0, 1]
    needs_pair_verification: float
    needs_external_tools: float
    source: str                       # "jev" | "heuristic_fallback" | "replay" | "fixed"
    answerable_from_inputs: float = 1.0  # P(answerable from supplied images + evidence)
    jev_model: Optional[str] = None
    request_id: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    latency_seconds: Optional[float] = None
    image_aware: bool = False
    error: Optional[str] = None       # why Jev was not used, when it was not
    raw: Optional[dict[str, Any]] = None
    # Answer-free features derived from the inputs (pair features, image count,
    # whether evidence was supplied). Used by the learned quality predictor.
    input_features: Optional[dict[str, Any]] = None

    def flags(self, threshold: float = 0.5) -> dict[str, bool]:
        return {
            "needs_image_retrieval": self.needs_image_retrieval >= threshold,
            "needs_pair_verification": self.needs_pair_verification >= threshold,
            "needs_external_tools": self.needs_external_tools >= threshold,
            "answerable_from_inputs": self.answerable_from_inputs >= threshold,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["flags"] = self.flags()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryClassification":
        data = {k: v for k, v in data.items() if k != "flags"}
        for key in ("query_type", "difficulty", "visual_expertise"):
            data[key] = LabelDistribution(**data[key])
        return cls(**data)


@dataclass
class CandidateScore:
    key: str
    model_id: str
    eligible: bool
    exclusion_reasons: list[str] = field(default_factory=list)
    expected_quality: Optional[float] = None      # normalized 0..1, before penalty
    unsupported_rate: Optional[float] = None
    adjusted_quality: Optional[float] = None      # after unsupported-claim (and, in accuracy mode, uncertainty) penalty
    quality_std: Optional[float] = None           # bootstrap std of the learned predictor, when used
    est_cost_usd: Optional[float] = None
    cost_source: Optional[str] = None
    est_latency_seconds: Optional[float] = None
    utility: Optional[float] = None
    score_sources: list[str] = field(default_factory=list)


@dataclass
class RoutingDecision:
    mode: str
    selected_key: Optional[str]
    selected_model_id: Optional[str]
    fallback_keys: list[str]
    rationale: list[str]
    candidates: list[CandidateScore]
    warnings: list[str] = field(default_factory=list)
    budget_usd: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VLMResult:
    key: str
    model_id: str
    backend: str
    status: str                       # "ok" | "error"
    answer: Optional[str] = None
    latency_seconds: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    cost_source: Optional[str] = None  # "provider_reported" | "estimated_from_registry_pricing"
    provider: Optional[str] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    raw_text: Optional[str] = None
    structured: Optional[dict[str, Any]] = None   # parsed JSON answer when the model followed the format

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
