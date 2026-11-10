from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.src.models import load_model_config
from benchmark.src.paths import (
    BUILDING_QUERY_RESULTS_PATH,
    PROJECT_ROOT,
    RESOLVED_MODELS_PATH,
)
from benchmark.src.providers.common import ProviderError
from benchmark.src.providers.openai_provider import generate as openai_generate
from benchmark.src.providers.openrouter_provider import generate as openrouter_generate
from benchmark.src.redact import sanitize_text

MODEL_KEY_BY_BACKEND = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "clough_true_match",
        "difficulty": "hard",
        "image_a": "data/clough/13C4114-P1-029.jpg",
        "image_b": "data/clough/26-R10410-P96-002.JPG",
        "expected_same_building": True,
        "reference_landmarks": {
            "image_a": "Clough Undergraduate Learning Commons interior",
            "image_b": "Clough Undergraduate Learning Commons exterior",
        },
        "reference_answer": (
            "Yes. Image A is an interior of the Clough Undergraduate Learning "
            "Commons, while Image B is the exterior of the same building. The "
            "large glazed volumes, exposed structural/ceiling language, and "
            "campus-facing building geometry provide the cross-view connection."
        ),
        "question": (
            "Do these two photos show the same building (possibly from a different "
            "viewpoint, distance, time of day, year, or with one photo taken inside "
            "and one outside)? Answer yes or no, then give careful visual reasoning. "
            "This is an adversarial cross-view test: identify the likely landmark or "
            "building shown in Image A and independently identify the likely landmark "
            "or building shown in Image B. Compare distinctive architecture, structure, "
            "materials, fenestration, circulation elements, and spatial context. Do not "
            "use filenames, paths, metadata, or assumptions from the prompt. If one "
            "image is interior and the other exterior, explain the specific evidence "
            "that connects them; do not claim a match merely because both could be on "
            "a university campus."
        ),
    },
    {
        "case_id": "tech_tower_vs_kessler_campanile_false_match",
        "difficulty": "hard",
        "image_a": "data/tech-tower/22C10400-P3-005.JPG",
        "image_b": "data/kessler-campanile/22C10400-P3-008.JPG",
        "expected_same_building": False,
        "reference_landmarks": {
            "image_a": "Georgia Tech Tech Tower",
            "image_b": "Georgia Tech Kessler Campanile",
        },
        "reference_answer": (
            "No. Image A is Georgia Tech's Tech Tower and Image B is Georgia Tech's "
            "Kessler Campanile. They are separate campus landmarks with different "
            "tower proportions, openings, roof details, and surrounding context."
        ),
        "question": (
            "Do these two photos show the same building (possibly from a different "
            "viewpoint, distance, time of day, year, or with one photo taken inside "
            "and one outside)? Answer yes or no even if the images are on the same "
            "campus or have a similar historic/vertical appearance. "
            "Give careful visual reasoning and explicitly identify the landmark in "
            "Image A and the landmark in Image B before comparing them. This is an "
            "adversarial negative pair: distinguish the two likely Georgia Tech "
            "campus landmarks using silhouette, tower proportions, "
            "roof or clock/bell details, masonry, openings, surrounding plaza, and "
            "adjacent buildings. Do not use filenames, paths, metadata, or assumptions "
            "from the prompt, and do not collapse two nearby campus landmarks into one "
            "just because they share an architectural style."
        ),
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_slots() -> list[dict[str, Any]]:
    if RESOLVED_MODELS_PATH.exists():
        rows = json.loads(RESOLVED_MODELS_PATH.read_text(encoding="utf-8"))
        if isinstance(rows, list):
            return rows
    slots = load_model_config().get("slots") or []
    return [
        {
            "logical_name": slot["logical_name"],
            "requested_model_id": slot.get("preferred_model_id"),
            "resolved_model_id": slot.get("preferred_model_id"),
            "backend": slot["backend"],
        }
        for slot in slots
    ]


def _settings() -> dict[str, Any]:
    return {
        "max_attempts": 3,
        "retry_backoff_seconds": [1, 2],
        "timeout_seconds": 180,
        "max_output_tokens": 900,
        "empty_output_retry_max_tokens": 1800,
        "temperature": 0,
    }


def _generate(slot: dict[str, Any], prompt: str, image_a: Path, image_b: Path) -> dict[str, Any]:
    model = slot.get("resolved_model_id") or slot.get("requested_model_id")
    if not model:
        raise ProviderError("model slot has no resolved or requested model id")
    backend = slot.get("backend")
    if backend == "openai":
        return openai_generate(model, prompt, image_a, image_b, _settings())
    if backend == "openrouter":
        return openrouter_generate(model, prompt, image_a, image_b, _settings())
    raise ProviderError(f"unsupported backend: {backend}")


def _run_case(case: dict[str, Any], slots: list[dict[str, Any]], dry_run: bool) -> dict[str, Any]:
    image_a = PROJECT_ROOT / case["image_a"]
    image_b = PROJECT_ROOT / case["image_b"]
    if not image_a.is_file() or not image_b.is_file():
        raise FileNotFoundError(f"missing image pair: {image_a}, {image_b}")

    result_case = {
        **case,
        "model_answers": [],
    }
    for slot in slots:
        backend = str(slot.get("backend") or "")
        model = slot.get("resolved_model_id") or slot.get("requested_model_id")
        answer: dict[str, Any] = {
            "logical_model_name": slot.get("logical_name"),
            "requested_model_id": slot.get("requested_model_id"),
            "resolved_model_id": model,
            "backend": backend,
            "status": "pending" if dry_run else "error",
            "answer": None,
            "latency_seconds": None,
            "input_tokens": None,
            "output_tokens": None,
            "reported_cost_usd": None,
            "error": None,
        }
        if dry_run:
            result_case["model_answers"].append(answer)
            continue
        key_name = MODEL_KEY_BY_BACKEND.get(backend)
        if key_name and not os.environ.get(key_name):
            answer["error"] = f"{key_name} is not set"
            result_case["model_answers"].append(answer)
            continue
        try:
            result = _generate(slot, case["question"], image_a, image_b)
            answer.update(
                {
                    "status": "ok",
                    "answer": result.get("response_text"),
                    "resolved_model_id": result.get("resolved_model_id") or model,
                    "latency_seconds": result.get("latency_seconds"),
                    "input_tokens": result.get("input_tokens"),
                    "output_tokens": result.get("output_tokens"),
                    "reported_cost_usd": result.get("reported_cost_usd"),
                    "parameter_adjustments": result.get("actual_parameter_adjustments") or [],
                }
            )
        except Exception as exc:
            answer["error"] = sanitize_text(str(exc))
        result_case["model_answers"].append(answer)
    return result_case


def run(dry_run: bool = False, model_names: set[str] | None = None) -> dict[str, Any]:
    slots = _load_slots()
    if model_names:
        slots = [slot for slot in slots if slot.get("logical_name") in model_names]
    payload = {
        "schema_version": "1.0",
        "created_at_utc": _utc_now(),
        "runner": "benchmark.src.run_building_queries",
        "purpose": "Same-building identification from interior/exterior and adversarial campus landmarks",
        "cases": [_run_case(case, slots, dry_run) for case in CASES],
    }
    BUILDING_QUERY_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUILDING_QUERY_RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hard same-building VLM queries.")
    parser.add_argument("--dry-run", action="store_true", help="Write the case/model manifest without API calls.")
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Logical model names to run, for example OPENAI_FRONTIER_VISION GEMINI.",
    )
    args = parser.parse_args()
    payload = run(dry_run=args.dry_run, model_names=set(args.models) if args.models else None)
    counts = {
        status: sum(
            1
            for case in payload["cases"]
            for answer in case["model_answers"]
            if answer["status"] == status
        )
        for status in ("ok", "error", "pending")
    }
    print(f"Wrote {BUILDING_QUERY_RESULTS_PATH}")
    print(f"Model answers: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
