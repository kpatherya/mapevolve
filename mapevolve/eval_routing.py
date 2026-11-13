"""Offline routing evaluation over a v2 profile (no network, no paid calls).

Reports, for one family and a representative request (query length, image
count): frontier coverage, a sweep over the tradeoff parameter, and savings
versus always-best and always-cheapest baselines. Optionally scores a
classifier against a labelled manifest (``[{"query": ..., "family": ...}]``).

Every number is computed from the recorded profile; nothing here measures a
model. Results are only as good as the profile (see ``require_coverage``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .contracts import RoutingRequest
from .jev_classifier import ClassificationResult, JevQueryClassifier, rules_classification
from .profile_builder import EvaluatedProfile, load_evaluated_profile
from .tradeoff_policy import (
    Candidate,
    CostAssumptions,
    NoEligibleModelError,
    TradeoffConfig,
    estimate_request_cost,
    pareto_candidates,
    select_candidate,
)


def routing_report(
    profile: EvaluatedProfile,
    family: str,
    request: Any,  # needs .query and .images only
    *,
    lams: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    use_lower_bound: bool = True,
    assumptions: CostAssumptions = CostAssumptions(),
) -> dict[str, Any]:
    aggregates, excluded = profile.family_aggregates(family)
    if not aggregates:
        raise ValueError(f"no evaluated models for family {family!r}")
    cands = [
        Candidate(
            a.model_id,
            a.accuracy_lower_bound if use_lower_bound else a.accuracy,
            estimate_request_cost(profile.models[a.model_id], request, assumptions),
            a,
        )
        for a in aggregates
    ]
    frontier = pareto_candidates(cands)
    best = max(cands, key=lambda c: (c.accuracy, -c.cost_usd))
    cheapest = min(cands, key=lambda c: (c.cost_usd, c.model_id))
    sweep = []
    for lam in lams:
        pick = select_candidate(frontier, TradeoffConfig(mode="utility", lam=lam, use_lower_bound=use_lower_bound))
        sweep.append(
            {
                "lam": lam,
                "model_id": pick.model_id,
                "accuracy": pick.accuracy,
                "cost_usd": pick.cost_usd,
                "savings_vs_always_best": 1.0 - pick.cost_usd / best.cost_usd if best.cost_usd else 0.0,
                "accuracy_gap_vs_best": best.accuracy - pick.accuracy,
                # Regret against the best model: how much accuracy was given up for the saving.
                "regret": best.accuracy - pick.accuracy,
            }
        )
    return {
        "family": family,
        "profile_version": profile.profile_version,
        "coverage": profile.coverage(family),
        "excluded_incomplete_coverage": list(excluded),
        "models_evaluated": len(cands),
        "frontier": [{"model_id": c.model_id, "accuracy": c.accuracy, "cost_usd": c.cost_usd} for c in frontier],
        "frontier_coverage": len(frontier) / len(cands),
        "always_best": {"model_id": best.model_id, "accuracy": best.accuracy, "cost_usd": best.cost_usd},
        "always_cheapest": {"model_id": cheapest.model_id, "accuracy": cheapest.accuracy, "cost_usd": cheapest.cost_usd},
        "sweep": sweep,
        "accuracy_basis": "wilson_lower_bound" if use_lower_bound else "pooled_mean",
    }


def classifier_report(
    manifest: list[dict[str, str]], classify: Callable[[RoutingRequest], ClassificationResult], image_paths: list[str]
) -> dict[str, Any]:
    correct, rows = 0, []
    for i, item in enumerate(manifest):
        result = classify(RoutingRequest(request_id=f"m{i}", query=item["query"], images=list(image_paths)))
        ok = result.interpretation.family == item["family"]
        correct += ok
        rows.append({"query": item["query"], "expected": item["family"],
                     "got": result.interpretation.family, "source": result.source, "ok": ok})
    return {"n": len(rows), "accuracy": correct / len(rows) if rows else None, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline routing evaluation (no network).")
    p.add_argument("--profile", required=True, help="schema v2 profile JSON")
    p.add_argument("--family", default="same_building_verification")
    p.add_argument("--query", default="Are these the same building?")
    p.add_argument("--images", type=int, default=2, help="image count of the representative request")
    p.add_argument("--mean-accuracy", action="store_true")
    p.add_argument("--classifier-manifest", help="JSON list of {query, family}")
    p.add_argument("--classifier", choices=("rules", "jev"), default="rules")
    p.add_argument("--image", action="append", default=[], help="existing image paths for classifier requests")
    args = p.parse_args(argv)
    try:
        profile = load_evaluated_profile(Path(args.profile))
        # Cost estimation only uses the image count, so placeholder paths are never read.
        request = SimpleNamespace(query=args.query, images=["<image>"] * args.images)
        report: dict[str, Any] = {
            "routing": routing_report(profile, args.family, request, use_lower_bound=not args.mean_accuracy)
        }
        if args.classifier_manifest:
            manifest = json.loads(Path(args.classifier_manifest).read_text(encoding="utf-8"))
            classify = JevQueryClassifier() if args.classifier == "jev" else rules_classification
            report["classifier"] = classifier_report(manifest, classify, args.image)
    except (ValueError, OSError, NoEligibleModelError) as error:
        print(f"eval_routing: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
