"""Build config/model_registry.json from static config, live OpenRouter
metadata, and the pilot benchmark results.

    python -m scripts.build_registry            # fetch current OpenRouter data
    python -m scripts.build_registry --offline  # keep pricing/availability null

Prices are never hardcoded: they come from https://openrouter.ai/api/v1/models
at build time and carry a `retrieved_at` timestamp. When a model is not listed,
its price stays null and the gap is recorded under `missing_data`.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from jev_router.benchmark_data import load_responses, model_metrics

ROOT = Path(__file__).resolve().parents[1]
MODELS_YAML = ROOT / "config" / "models.yaml"
OUTPUT = ROOT / "config" / "model_registry.json"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

GLOBAL_LIMITATIONS = [
    "Evaluation scores come from a 6-case pilot (2 easy, 2 moderate, 2 hard); "
    "each model x difficulty cell has n=2. Treat rankings as preliminary.",
    "Scores are GPT-5.1 LLM-judge scores (0-4). Human-adjudication columns in "
    "benchmark/artifacts/manual_review.csv are empty; filled human scores override "
    "judge scores automatically on the next build.",
    "GPT-5.1 judged its own 6 responses (self_judged=True in judged_responses.csv).",
    "Only temporal_comparison (two-image 'what changed?') has graded results. "
    "landmark_identification, cross_view_matching, spatial_reasoning and "
    "factual_retrieval have no task-specific scores; routing falls back to the "
    "general score for them.",
    "One response per model and case at temperature 0; no repetitions, so no "
    "variance estimates.",
    "Latency was measured from one client location in October 2026 and includes "
    "provider queueing.",
]


def fetch_openrouter_models(timeout: float = 30) -> dict[str, dict[str, Any]]:
    request = urllib.request.Request(OPENROUTER_MODELS_URL, headers={"User-Agent": "jev-router-registry"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))["data"]
    return {m["id"]: m for m in data}


def _per_mtok(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    return round(float(value) * 1_000_000, 6)


def build_entry(
    spec: dict[str, Any],
    listing: Optional[dict[str, Any]],
    metrics: Optional[dict[str, Any]],
    retrieved_at: Optional[str],
    offline: bool,
) -> dict[str, Any]:
    missing: list[str] = []
    limitations: list[str] = []
    lookup_id = spec.get("pricing_reference_openrouter_id") or spec["model_id"]

    if offline:
        availability = {"status": "unknown", "checked_at": None, "note": "built with --offline"}
        pricing = None
        missing.append("pricing (offline build)")
    elif listing is None:
        availability = {
            "status": "not_listed",
            "checked_at": retrieved_at,
            "note": f"{lookup_id} is not in the current OpenRouter model list.",
        }
        pricing = None
        missing.append(f"current pricing: {lookup_id} not listed on OpenRouter")
    else:
        expiration = listing.get("expiration_date")
        availability = {
            "status": "expiring" if expiration else "listed",
            "checked_at": retrieved_at,
            "expiration_date": expiration,
        }
        p = listing.get("pricing", {})
        pricing = {
            "input_usd_per_mtok": _per_mtok(p.get("prompt")),
            "output_usd_per_mtok": _per_mtok(p.get("completion")),
            "image_usd_per_image": float(p["image"]) if p.get("image") not in (None, "", "0") else None,
            "source": "openrouter_models_api",
            "source_model_id": lookup_id,
            "retrieved_at": retrieved_at,
        }
        if spec["backend"] == "openai":
            pricing["note"] = (
                "Reference price from the OpenRouter listing of the same model; direct "
                "OpenAI API pricing is not exposed programmatically. Verify before billing."
            )
        if expiration:
            limitations.append(f"OpenRouter lists an expiration date of {expiration}.")

    modalities = (listing or {}).get("architecture", {}).get("input_modalities")
    image_input = {
        "supported": (("image" in modalities) if modalities else None),
        "listed_input_modalities": modalities,
        "benchmark_two_image_smoke_test": spec.get("benchmark_two_image_smoke_test"),
        "max_images": spec.get("max_images", 1),
    }
    if image_input["supported"] is None:
        # Fall back to the benchmark's live two-image smoke test result.
        image_input["supported"] = spec.get("benchmark_two_image_smoke_test") == "pass"
        image_input["source"] = "benchmark_smoke_test_only"
    else:
        image_input["source"] = "openrouter_modalities"

    if metrics is None:
        missing.append("all evaluation metrics (model not in judged_responses.csv)")
        evaluation = None
        latency = tokens = observed_cost = None
    else:
        evaluation = {
            "general": metrics["general"],
            "unsupported_claim_rate": {
                "rate": metrics["general"]["unsupported_claim_rate"],
                "n": metrics["general"]["n"],
                "definition": "share of responses asserting a physical/structural change the answer key does not support",
            },
            "task_scores": metrics["task_scores"],
            "score_sources": metrics["score_sources"],
            "judge_model": "gpt-5.1-2025-11-13",
        }
        latency = {**metrics["latency"], "source": "benchmark pilot (Oct 2026)"}
        tokens = metrics["tokens"]
        observed_cost = metrics["observed_cost"]
        if observed_cost["mean_usd_per_query"] is None:
            missing.append("observed cost (provider did not report cost in the benchmark)")
        for task in ("landmark_identification", "cross_view_matching", "spatial_reasoning", "factual_retrieval"):
            missing.append(f"task score: {task}")

    if spec.get("notes"):
        limitations.append(" ".join(spec["notes"].split()))

    return {
        "key": spec["key"],
        "display_name": spec.get("display_name", spec["key"]),
        "model_id": spec["model_id"],
        "resolved_model_id": spec.get("resolved_model_id", spec["model_id"]),
        "provider": spec["provider"],
        "backend": spec["backend"],
        "max_output_tokens": spec.get("max_output_tokens"),
        "chat_kwargs": spec.get("chat_kwargs") or {},
        "availability": availability,
        "image_input": image_input,
        "pricing": pricing,
        "latency": latency,
        "tokens": tokens,
        "observed_cost": observed_cost,
        "evaluation": evaluation,
        "missing_data": missing,
        "limitations": limitations,
    }


def build(offline: bool = False) -> dict[str, Any]:
    specs = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))["models"]
    listings: dict[str, dict[str, Any]] = {}
    retrieved_at = None
    if not offline:
        listings = fetch_openrouter_models()
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics = model_metrics(load_responses())
    models = []
    for spec in specs:
        lookup = spec.get("pricing_reference_openrouter_id") or spec["model_id"]
        models.append(build_entry(spec, listings.get(lookup), metrics.get(spec["key"]), retrieved_at, offline))
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "static_config": "config/models.yaml",
            "pricing_and_modalities": OPENROUTER_MODELS_URL if not offline else None,
            "evaluation": "benchmark/artifacts/judged_responses.csv (+ manual_review.csv overrides)",
            "tokens": "benchmark/runs/raw_responses.jsonl",
        },
        "limitations": GLOBAL_LIMITATIONS,
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="Skip the OpenRouter fetch")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        registry = build(offline=args.offline)
    except OSError as exc:
        print(f"error: could not fetch OpenRouter models ({exc}); retry or use --offline", file=sys.stderr)
        return 1
    args.output.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    for m in registry["models"]:
        price = m["pricing"] or {}
        print(
            f"{m['key']:<24} {m['availability']['status']:<11} "
            f"in=${price.get('input_usd_per_mtok')}/M out=${price.get('output_usd_per_mtok')}/M "
            f"score={m['evaluation']['general']['mean_score'] if m['evaluation'] else None}"
        )
    print(f"wrote {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
