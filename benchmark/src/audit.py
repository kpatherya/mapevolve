from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from benchmark.src.aggregate_results import build_joined
from benchmark.src.make_reports import figure_source
from benchmark.src.paths import (
    ARTIFACTS_DIR,
    BENCHMARK_ROOT,
    CASE_IDS,
    CONTACT_SHEET_PATH,
    FIGURE_DATA_PATH,
    INTEGRITY_PATH,
    JOINED_RESULTS_PATH,
    LOGICAL_MODEL_ORDER,
    MANIFEST_PATH,
    PREPROCESS_LOG_PATH,
    RAW_API_DIR,
    RESOLVED_MODELS_PATH,
)
from benchmark.src.prepare_data import prompt_contains_forbidden_metadata, sha256_file
from benchmark.src.prepare_data import load_cases
from benchmark.src.prompting import render_judge_prompt, render_vlm_prompt
from benchmark.src.scoring import compute_totals
from benchmark.src.validate_data import validate

SECRET_RE = re.compile(r"sk-[A-Za-z0-9]{20,}|AQ\.[A-Za-z0-9_\-]{20,}")
ALLOWED_OPS = {
    "exif_transpose",
    "convert_rgb",
    "resize_long_edge_if_needed",
    "strip_metadata",
    "jpeg_q95",
}
FIGURES = (
    "model_case_score_matrix.png",
    "mean_score_by_difficulty.png",
    "hallucination_rate_by_model.png",
    "latency_by_model.png",
)

def _check_secrets(errors: list[str]) -> None:
    roots = [ARTIFACTS_DIR, BENCHMARK_ROOT / "runs", BENCHMARK_ROOT / "prompts", BENCHMARK_ROOT / "config"]
    files = [BENCHMARK_ROOT / "cases.json", BENCHMARK_ROOT / "README.md"]
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    for path in files:
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if SECRET_RE.search(text):
            errors.append(f"possible API key text in {path.relative_to(BENCHMARK_ROOT)}")

def _sidecar_prompts(errors: list[str], cases: dict[str, dict[str, Any]]) -> None:
    if not RAW_API_DIR.exists():
        errors.append("raw API directory is missing")
        return
    for path in RAW_API_DIR.glob("*.json"):
        if path.name.startswith("judge_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        parts = payload.get("text_parts") or []
        if payload.get("tools") not in (None, [], False):
            errors.append(f"{path.name} enabled tools")
        if payload.get("web_search") not in (False, None):
            errors.append(f"{path.name} enabled web search")
        if payload.get("retrieval") not in (False, None):
            errors.append(f"{path.name} enabled retrieval")
        blob = json.dumps(payload)
        if "data:image" in blob or "preflight_contact_sheet" in blob:
            errors.append(f"{path.name} contains image data or the contact sheet")
        if len(parts) >= 3:
            prompt = parts[-1]
            case_id = path.stem.split("_")[-1]
            case = cases.get(case_id)
            if case is not None:
                expected = render_vlm_prompt(case["question"])
                if prompt != expected:
                    errors.append(f"{path.name} prompt does not match the fixed template")
                problems = prompt_contains_forbidden_metadata(prompt, case)
                if problems:
                    errors.append(f"{path.name} prompt exposes {problems}")
                if case["ground_truth_raw"] in prompt:
                    errors.append(f"{path.name} includes the raw ground truth")

def _judge_sidecars(errors: list[str], cases: dict[str, dict[str, Any]], joined: list[dict[str, Any]]) -> None:
    for row in joined:
        if row.get("status") != "success" or row.get("judge_status") != "success":
            continue
        anon = row.get("anonymous_response_id")
        path = RAW_API_DIR / f"judge_{anon}.json"
        if not path.exists():
            errors.append(f"missing judge sidecar for {row.get('logical_model_name')} {row.get('case_id')}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("saw_images") is not False or payload.get("image_count") != 0:
            errors.append(f"{path.name} does not record a text-only judge call")
        if "data:image" in json.dumps(payload):
            errors.append(f"{path.name} contains image data")
        case = cases[row["case_id"]]
        expected = render_judge_prompt(
            case["question"],
            case["ground_truth_raw"],
            case["rubric"],
            row.get("response_text") or "",
        )
        if payload.get("prompt") != expected:
            errors.append(f"{path.name} judge prompt was not built only from question, ground truth, rubric, and response")
        raw_total, final_total, label = compute_totals(
            int(row["core_change_score"]),
            int(row["confounder_score"]),
            int(row["hallucination_free_score"]),
            bool(row["hallucinated_change"]),
        )
        if raw_total != row["raw_total"] or final_total != row["final_total"] or label != row["label"]:
            errors.append(f"score mismatch for {row.get('logical_model_name')} {row.get('case_id')}")
        if row["raw_total"] != row["core_change_score"] + row["confounder_score"] + row["hallucination_free_score"]:
            errors.append(f"raw total does not sum for {row.get('logical_model_name')} {row.get('case_id')}")

def _figures(errors: list[str]) -> None:
    if not FIGURE_DATA_PATH.exists():
        errors.append("figure_data.json is missing")
        return
    saved = json.loads(FIGURE_DATA_PATH.read_text(encoding="utf-8"))
    fresh = figure_source()
    if saved != fresh:
        errors.append("figure_data.json does not match the CSV tables")
    for name in FIGURES:
        path = ARTIFACTS_DIR / name
        if not path.exists() or path.stat().st_size < 1000:
            errors.append(f"figure missing or empty: {name}")

def audit() -> dict[str, Any]:
    errors: list[str] = []
    errors.extend(validate())
    cases = {case["case_id"]: case for case in load_cases()}
    if len(cases) != 6:
        errors.append(f"expected 6 cases, found {len(cases)}")
    joined = build_joined() if (BENCHMARK_ROOT / "runs" / "raw_responses.jsonl").exists() else []
    successes = [row for row in joined if row.get("status") == "success"]
    failures = [row for row in joined if row.get("status") != "success"]
    slots = json.loads(RESOLVED_MODELS_PATH.read_text(encoding="utf-8")) if RESOLVED_MODELS_PATH.exists() else []
    if [slot.get("logical_name") for slot in slots] != list(LOGICAL_MODEL_ORDER):
        errors.append("resolved model slots are not the required eight, in order")
    unresolved = [slot.get("logical_name") for slot in slots if not slot.get("resolved_model_id")]
    if unresolved:
        errors.append(f"unresolved slots: {unresolved}")
    if len(successes) != 48:
        errors.append(f"successful responses {len(successes)} of 48")
        for row in failures:
            errors.append(f"failed {row.get('logical_model_name')} {row.get('case_id')}: {row.get('error')}")
    for row in successes:
        if not row.get("resolved_model_id"):
            errors.append(f"{row.get('logical_model_name')} {row.get('case_id')} has no resolved model id")
        if row.get("tools_enabled") not in (False, None) or row.get("web_search") not in (False, None):
            errors.append(f"{row.get('logical_model_name')} {row.get('case_id')} enabled tools or web")
        if row.get("judge_status") != "success":
            errors.append(f"missing judge result for {row.get('logical_model_name')} {row.get('case_id')}")
    _sidecar_prompts(errors, cases)
    _judge_sidecars(errors, cases, joined)
    if not MANIFEST_PATH.exists():
        errors.append("run manifest is missing")
    else:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if not manifest.get("prompt_sha256") or not manifest.get("cases_sha256"):
            errors.append("manifest is missing prompt or cases hashes")
        if len(manifest.get("images") or []) != 12:
            errors.append("manifest does not store 12 image hash pairs")
        for image in manifest.get("images") or []:
            if not image.get("original_sha256") or not image.get("processed_sha256"):
                errors.append(f"missing hash for {image.get('case_id')} {image.get('side')}")
    if PREPROCESS_LOG_PATH.exists():
        log = json.loads(PREPROCESS_LOG_PATH.read_text(encoding="utf-8"))
        for item in log:
            ops = set(item.get("operations") or [])
            if not ops <= ALLOWED_OPS:
                errors.append(f"unexpected image operation on {item.get('case_id')} {item.get('side')}: {ops - ALLOWED_OPS}")
            if item.get("upscaled"):
                errors.append(f"{item.get('case_id')} {item.get('side')} was upscaled")
    else:
        errors.append("preprocessing log is missing")
    if not CONTACT_SHEET_PATH.exists():
        errors.append("contact sheet is missing")
    expected_processed = {BENCHMARK_ROOT / "data" / "processed" / f"{case_id}_{side}.jpg" for case_id in CASE_IDS for side in ("A", "B")}
    if {path for path in expected_processed if path.exists()} != expected_processed:
        errors.append("processed image set is not the 12 neutral JPEGs")
    for case in cases.values():
        for key in ("image_a_sha256", "image_b_sha256", "original_image_a_sha256", "original_image_b_sha256"):
            if not case.get(key):
                errors.append(f"{case['case_id']} missing {key}")
        processed = BENCHMARK_ROOT / case["image_a"]
        if processed.exists() and sha256_file(processed) != case["image_a_sha256"]:
            errors.append(f"{case['case_id']} image A hash drifted")
    _figures(errors)
    _check_secrets(errors)
    report = {
        "passed": not errors,
        "successful_responses": len(successes),
        "intended_responses": 48,
        "failures": [
            {
                "logical_model_name": row.get("logical_model_name"),
                "case_id": row.get("case_id"),
                "error": row.get("error"),
            }
            for row in failures
        ],
        "errors": errors,
    }
    INTEGRITY_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

def main() -> None:
    if not JOINED_RESULTS_PATH.exists():
        print("joined_results.json is missing; audit will still check prepared data.")
    report = audit()
    print(f"Integrity passed: {report['passed']}")
    print(f"Successful responses: {report['successful_responses']}/48")
    for error in report["errors"]:
        print(f"- {error}")
    if not report["passed"]:
        sys.exit(1)

if __name__ == "__main__":
    main()
