"""Evaluation-backed model profiles (schema v2).

A v2 profile stores per-benchmark results with sample counts instead of one
hand-entered accuracy. Family accuracy is aggregated with an explicit rule:

* only benchmarks whose metric is ``accuracy`` (a proportion) are pooled;
* scores are pooled by sample count (``sum(score * n) / sum(n)``);
* a model is eligible for a family only if it has a result on every benchmark
  recorded for that family (otherwise pooled scores are not comparable);
* a 95% Wilson lower bound is kept as the conservative estimate.

Nothing here runs a model; it only validates and aggregates recorded results.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import FAMILIES
from .pareto_policy import ProfileValidationError

PROFILES_V2_SCHEMA_VERSION = 2
_WILSON_Z = 1.96
_BENCHMARK_FIELDS = {"task", "metric", "split", "source", "date", "notes"}
_BENCHMARK_REQUIRED = {"task", "metric", "split", "source", "date"}
_MODEL_FIELDS = {
    "tier",
    "license",
    "architecture",
    "input_cost_usd_per_1m_tokens",
    "output_cost_usd_per_1m_tokens",
    "tokens_per_image",
    "evaluated_as",
}
_MODEL_REQUIRED = {
    "tier",
    "license",
    "input_cost_usd_per_1m_tokens",
    "output_cost_usd_per_1m_tokens",
    "tokens_per_image",
}
_RESULT_FIELDS = {"family", "benchmark", "model_id", "score", "n"}


def _num(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileValidationError(f"{name} must be a number")
    return float(value)


def _check_keys(payload: Any, name: str, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProfileValidationError(f"{name} must be an object")
    unknown = set(payload) - allowed
    if unknown:
        raise ProfileValidationError(f"{name} has unknown fields: {sorted(unknown)}")
    missing = required - set(payload)
    if missing:
        raise ProfileValidationError(f"{name} is missing fields: {sorted(missing)}")
    return payload


def wilson_lower_bound(p: float, n: int, z: float = _WILSON_Z) -> float:
    if n <= 0:
        return 0.0
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    task: str
    metric: str
    split: str
    source: str
    date: str


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    tier: str
    license: str
    input_cost_usd_per_1m_tokens: float
    output_cost_usd_per_1m_tokens: float
    tokens_per_image: int
    architecture: str | None = None
    evaluated_as: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    family: str
    benchmark: str
    model_id: str
    score: float
    n: int


@dataclass(frozen=True)
class FamilyAggregate:
    model_id: str
    accuracy: float
    accuracy_lower_bound: float
    n: int
    benchmarks: tuple[str, ...]


@dataclass(frozen=True)
class EvaluatedProfile:
    profile_version: str
    benchmarks: dict[str, BenchmarkSpec]
    models: dict[str, ModelSpec]
    results: tuple[BenchmarkResult, ...]

    def family_aggregates(self, family: str) -> tuple[tuple[FamilyAggregate, ...], tuple[str, ...]]:
        """Return (eligible aggregates, model ids excluded for incomplete coverage)."""
        if family not in FAMILIES:
            raise ProfileValidationError(f"unsupported family: {family!r}")
        rows = [r for r in self.results if r.family == family]
        family_benchmarks = sorted({r.benchmark for r in rows})
        by_model: dict[str, dict[str, BenchmarkResult]] = {}
        for row in rows:
            by_model.setdefault(row.model_id, {})[row.benchmark] = row
        aggregates: list[FamilyAggregate] = []
        excluded: list[str] = []
        for model_id, per_benchmark in sorted(by_model.items()):
            if set(per_benchmark) != set(family_benchmarks):
                excluded.append(model_id)
                continue
            total_n = sum(r.n for r in per_benchmark.values())
            pooled = sum(r.score * r.n for r in per_benchmark.values()) / total_n
            aggregates.append(
                FamilyAggregate(
                    model_id=model_id,
                    accuracy=pooled,
                    accuracy_lower_bound=wilson_lower_bound(pooled, total_n),
                    n=total_n,
                    benchmarks=tuple(family_benchmarks),
                )
            )
        return tuple(aggregates), tuple(excluded)

    def coverage(self, family: str) -> dict[str, Any]:
        aggregates, excluded = self.family_aggregates(family)
        benchmarks = sorted({r.benchmark for r in self.results if r.family == family})
        return {
            "family": family,
            "eligible_models": len(aggregates),
            "excluded_models": list(excluded),
            "benchmarks": benchmarks,
        }

    def require_coverage(self, family: str, *, min_models: int = 10, min_benchmarks: int = 3) -> None:
        """Production gate: fail loudly if the profile is too thin to claim a Pareto curve."""
        info = self.coverage(family)
        problems = []
        if info["eligible_models"] < min_models:
            problems.append(f"{info['eligible_models']} eligible models < {min_models}")
        if len(info["benchmarks"]) < min_benchmarks:
            problems.append(f"{len(info['benchmarks'])} benchmarks < {min_benchmarks}")
        if problems:
            raise ProfileValidationError(f"{family} coverage insufficient: " + "; ".join(problems))


def build_profile(payload: Any) -> EvaluatedProfile:
    root = _check_keys(
        payload,
        "profile",
        {"schema_version", "profile_version", "benchmarks", "models", "results"},
        {"schema_version", "profile_version", "benchmarks", "models", "results"},
    )
    if root["schema_version"] != PROFILES_V2_SCHEMA_VERSION:
        raise ProfileValidationError(f"schema_version must be {PROFILES_V2_SCHEMA_VERSION}")
    version = root["profile_version"]
    if not isinstance(version, str) or not version.strip():
        raise ProfileValidationError("profile_version must be a non-empty string")

    if not isinstance(root["benchmarks"], dict) or not isinstance(root["models"], dict):
        raise ProfileValidationError("benchmarks and models must be objects")
    benchmarks: dict[str, BenchmarkSpec] = {}
    for bid, spec in root["benchmarks"].items():
        spec = _check_keys(spec, f"benchmarks.{bid}", _BENCHMARK_FIELDS, _BENCHMARK_REQUIRED)
        if spec["metric"] != "accuracy":
            raise ProfileValidationError(
                f"benchmarks.{bid}.metric={spec['metric']!r} cannot be pooled; "
                "only 'accuracy' proportions have an aggregation rule"
            )
        benchmarks[bid] = BenchmarkSpec(
            bid, spec["task"], spec["metric"], spec["split"], spec["source"], spec["date"]
        )

    models: dict[str, ModelSpec] = {}
    for mid, spec in root["models"].items():
        spec = _check_keys(spec, f"models.{mid}", _MODEL_FIELDS, _MODEL_REQUIRED)
        in_rate = _num(spec["input_cost_usd_per_1m_tokens"], f"models.{mid}.input_cost")
        out_rate = _num(spec["output_cost_usd_per_1m_tokens"], f"models.{mid}.output_cost")
        tpi = _num(spec["tokens_per_image"], f"models.{mid}.tokens_per_image")
        if in_rate < 0 or out_rate < 0 or tpi < 0:
            raise ProfileValidationError(f"models.{mid} costs and tokens_per_image must be >= 0")
        models[mid] = ModelSpec(
            mid,
            spec["tier"],
            spec["license"],
            in_rate,
            out_rate,
            int(tpi),
            spec.get("architecture"),
            spec.get("evaluated_as"),
        )

    if not isinstance(root["results"], list):
        raise ProfileValidationError("results must be a list")
    results: list[BenchmarkResult] = []
    seen: set[tuple[str, str, str]] = set()
    for i, raw in enumerate(root["results"]):
        raw = _check_keys(raw, f"results[{i}]", _RESULT_FIELDS, _RESULT_FIELDS)
        if raw["family"] not in FAMILIES:
            raise ProfileValidationError(f"results[{i}].family is not a known family")
        if raw["benchmark"] not in benchmarks:
            raise ProfileValidationError(f"results[{i}] references unknown benchmark {raw['benchmark']!r}")
        if raw["model_id"] not in models:
            raise ProfileValidationError(f"results[{i}] references unknown model {raw['model_id']!r}")
        score = _num(raw["score"], f"results[{i}].score")
        n = raw["n"]
        if not 0.0 <= score <= 1.0:
            raise ProfileValidationError(f"results[{i}].score must be in [0, 1]")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ProfileValidationError(f"results[{i}].n must be a positive integer")
        key = (raw["family"], raw["benchmark"], raw["model_id"])
        if key in seen:
            raise ProfileValidationError(f"duplicate result for {key}")
        seen.add(key)
        results.append(BenchmarkResult(raw["family"], raw["benchmark"], raw["model_id"], score, n))
    return EvaluatedProfile(version, benchmarks, models, tuple(results))


def load_evaluated_profile(path: Path) -> EvaluatedProfile:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ProfileValidationError(f"profile file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ProfileValidationError(f"profile file JSON decode failed: {error}") from error
    return build_profile(payload)


def is_evaluated_profile_file(path: Path) -> bool:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("schema_version") == PROFILES_V2_SCHEMA_VERSION
