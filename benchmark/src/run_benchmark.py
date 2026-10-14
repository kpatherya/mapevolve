from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import yaml

from benchmark.src.paths import (
    BENCHMARK_ROOT,
    BENCHMARK_YAML,
    CASES_PATH,
    LOGICAL_MODEL_ORDER,
    MANIFEST_PATH,
    RAW_API_DIR,
    RESOLVED_MODELS_PATH,
    RESPONSES_PATH,
    SMOKE_MD,
    VLM_PROMPT_PATH,
)
from benchmark.src.preflight import load_settings, resolve_slots
from benchmark.src.prepare_data import load_cases, sha256_file
from benchmark.src.prompting import render_vlm_prompt
from benchmark.src.providers.common import ProviderError
from benchmark.src.providers.openai_provider import generate as openai_generate
from benchmark.src.providers.openrouter_provider import generate as openrouter_generate
from benchmark.src.records import append_jsonl, new_run_id, read_jsonl, response_key, successful_pairs, utc_now
from benchmark.src.redact import sanitize_text
from benchmark.src.validate_data import validate

PROMPT_VERSION = "vlm_prompt_v1"

def _package_versions() -> dict[str, Optional[str]]:
    versions = {}
    for name in ("openai", "pillow", "PyYAML", "matplotlib", "httpx", "pytest"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions

def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=BENCHMARK_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

def _prompt_sha256() -> str:
    return hashlib.sha256(VLM_PROMPT_PATH.read_bytes()).hexdigest()

def _load_resolved(refresh: bool) -> list[dict[str, Any]]:
    if RESOLVED_MODELS_PATH.exists() and not refresh:
        return json.loads(RESOLVED_MODELS_PATH.read_text(encoding="utf-8"))
    print("Running model availability preflight.", flush=True)
    return resolve_slots(load_settings())

def _run_identity() -> tuple[str, str]:
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing.get("run_id"):
            return existing["run_id"], existing.get("created_at_utc") or utc_now()
    return new_run_id(), utc_now()

def write_manifest(run_id: str, created_at: str, cases: list[dict[str, Any]], models: list[dict[str, Any]]) -> None:
    settings = yaml.safe_load(BENCHMARK_YAML.read_text(encoding="utf-8"))
    responses = read_jsonl(RESPONSES_PATH)
    adjustments = []
    for row in responses:
        for note in row.get("actual_parameter_adjustments") or []:
            adjustments.append(
                {
                    "logical_model_name": row.get("logical_model_name"),
                    "case_id": row.get("case_id"),
                    "adjustment": note,
                }
            )
    images = []
    for case in cases:
        for side, image_key, original_key in (
            ("A", "image_a_sha256", "original_image_a_sha256"),
            ("B", "image_b_sha256", "original_image_b_sha256"),
        ):
            images.append(
                {
                    "case_id": case["case_id"],
                    "side": side,
                    "processed_sha256": case[image_key],
                    "original_sha256": case[original_key],
                }
            )
    payload = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "updated_at_utc": utc_now(),
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "package_versions": _package_versions(),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": _prompt_sha256(),
        "cases_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
        "images": images,
        "resolved_models": [
            {
                "logical_name": row.get("logical_name"),
                "requested_model_id": row.get("requested_model_id"),
                "resolved_model_id": row.get("resolved_model_id"),
                "backend": row.get("backend"),
                "provider_if_available": row.get("provider_if_available"),
                "status": row.get("status"),
                "substitution_reason": row.get("substitution_reason"),
            }
            for row in models
        ],
        "parameter_adjustments": adjustments,
        "benchmark_settings": settings,
        "tools_enabled": False,
        "web_search": False,
        "retrieval": False,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

def _base_record(run_id: str, case: dict[str, Any], slot: dict[str, Any], prompt: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "case_id": case["case_id"],
        "difficulty": case["difficulty"],
        "logical_model_name": slot["logical_name"],
        "requested_model_id": slot.get("requested_model_id"),
        "resolved_model_id": slot.get("resolved_model_id"),
        "backend": slot.get("backend"),
        "provider_if_available": slot.get("provider_if_available"),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "image_a_sha256": case["image_a_sha256"],
        "image_b_sha256": case["image_b_sha256"],
        "response_text": "",
        "response_sha256": None,
        "latency_seconds": None,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens_if_reported": None,
        "reported_cost_usd_if_available": None,
        "temperature_requested": 0,
        "actual_parameter_adjustments": [],
        "timestamp_utc": utc_now(),
        "status": "error",
        "tools_enabled": False,
        "web_search": False,
        "retrieval": False,
        "error": None,
    }

def _write_sidecar(record: dict[str, Any], prompt: str, result: Optional[dict[str, Any]]) -> None:
    RAW_API_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_API_DIR / f"{record['logical_model_name']}_{record['case_id']}.json"
    parameters = {} if result is None else result.get("parameters_sent") or {}
    payload = {
        "text_parts": ["Image A:", "Image B:", prompt],
        "image_a_sha256": record["image_a_sha256"],
        "image_b_sha256": record["image_b_sha256"],
        "detail": "high",
        "tools": None,
        "web_search": False,
        "retrieval": False,
        "parameters_sent": parameters,
        "adjustments": record.get("actual_parameter_adjustments") or [],
        "response": None if result is None else result.get("raw_response"),
        "error": record.get("error"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def execute_task(run_id: str, case: dict[str, Any], slot: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    prompt = render_vlm_prompt(case["question"])
    record = _base_record(run_id, case, slot, prompt)
    if not slot.get("resolved_model_id"):
        record["error"] = slot.get("substitution_reason") or "model slot unavailable"
        _write_sidecar(record, prompt, None)
        return record
    image_a = BENCHMARK_ROOT / case["image_a"]
    image_b = BENCHMARK_ROOT / case["image_b"]
    if sha256_file(image_a) != case["image_a_sha256"] or sha256_file(image_b) != case["image_b_sha256"]:
        record["error"] = "processed image hash does not match cases.json"
        _write_sidecar(record, prompt, None)
        return record
    try:
        if slot["backend"] == "openai":
            result = openai_generate(slot["resolved_model_id"], prompt, image_a, image_b, settings)
        elif slot["backend"] == "openrouter":
            result = openrouter_generate(slot["resolved_model_id"], prompt, image_a, image_b, settings)
        else:
            raise ProviderError(f"unknown backend {slot['backend']}")
    except Exception as exc:
        record["error"] = sanitize_text(str(exc))
        record["timestamp_utc"] = utc_now()
        _write_sidecar(record, prompt, None)
        return record
    text = result.get("response_text") or ""
    record.update(
        {
            "resolved_model_id": result.get("resolved_model_id") or slot.get("resolved_model_id"),
            "provider_if_available": result.get("provider_name") or slot.get("provider_if_available"),
            "response_text": text,
            "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "latency_seconds": result.get("latency_seconds"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "reasoning_tokens_if_reported": result.get("reasoning_tokens"),
            "reported_cost_usd_if_available": result.get("reported_cost_usd"),
            "temperature_requested": result.get("temperature_requested", 0),
            "actual_parameter_adjustments": result.get("actual_parameter_adjustments") or [],
            "timestamp_utc": utc_now(),
            "status": "success",
            "tools_enabled": False,
            "web_search": False,
            "retrieval": False,
            "error": None,
        }
    )
    _write_sidecar(record, prompt, result)
    return record

def write_smoke_report(models: list[dict[str, Any]]) -> None:
    rows = read_jsonl(RESPONSES_PATH)
    latest = {}
    for row in rows:
        if row.get("case_id") == "E1":
            latest[row.get("logical_model_name")] = row
    lines = [
        "# Smoke test",
        "",
        "E1 uses the fixed benchmark prompt. The earlier two-image check only asked for the word OK and is not scored.",
        "",
        "## Availability",
        "",
        "| Slot | Requested | Resolved | Backend | Image | Two-image smoke | Status | Substitution |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for slot in models:
        lines.append(
            "| {logical} | `{requested}` | `{resolved}` | {backend} | {image} | {smoke} | {status} | {reason} |".format(
                logical=slot.get("logical_name"),
                requested=slot.get("requested_model_id") or "",
                resolved=slot.get("resolved_model_id") or "",
                backend=slot.get("backend") or "",
                image=slot.get("image_input_supported"),
                smoke=slot.get("two_image_smoke_test"),
                status=slot.get("status"),
                reason=(slot.get("substitution_reason") or "").replace("|", "/"),
            )
        )
    lines.extend(["", "## E1 responses", ""])
    for name in LOGICAL_MODEL_ORDER:
        row = latest.get(name)
        lines.append(f"### {name}")
        if row is None:
            lines.append("")
            lines.append("No E1 response recorded.")
            lines.append("")
            continue
        lines.append("")
        lines.append(f"- Status: {row.get('status')}")
        lines.append(f"- Resolved model: `{row.get('resolved_model_id')}`")
        lines.append(f"- Latency seconds: {row.get('latency_seconds')}")
        lines.append(f"- Reported cost USD: {row.get('reported_cost_usd_if_available')}")
        adjustments = row.get("actual_parameter_adjustments") or []
        lines.append(f"- Parameter adjustments: {adjustments if adjustments else 'none'}")
        if row.get("error"):
            lines.append(f"- Error: {row.get('error')}")
        lines.append("")
        lines.append(row.get("response_text") or "")
        lines.append("")
    SMOKE_MD.write_text("\n".join(lines), encoding="utf-8")

def _select_names(raw: Optional[str], allowed: tuple[str, ...]) -> list[str]:
    if not raw:
        return list(allowed)
    chosen = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in chosen if item not in allowed]
    if unknown:
        raise SystemExit(f"Unknown selection {unknown}. Expected {list(allowed)}")
    return chosen

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MapEvolve pilot VLM benchmark.")
    parser.add_argument("--resume", action="store_true", help="Skip model/case pairs that already succeeded.")
    parser.add_argument("--force", action="store_true", help="Rerun selected pairs even if a success exists.")
    parser.add_argument("--models", default="", help="Comma-separated logical model names.")
    parser.add_argument("--cases", default="", help="Comma-separated case ids.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the plan without API calls.")
    parser.add_argument("--smoke-test", action="store_true", help="Record a smoke report. Defaults to case E1.")
    parser.add_argument("--refresh-models", action="store_true", help="Repeat model resolution and two-image smoke checks.")
    parser.add_argument("--workers", type=int, default=0, help="Parallel requests. Default comes from benchmark.yaml.")
    args = parser.parse_args()

    errors = validate()
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        print("Stopped before any model API call.")
        raise SystemExit(2)

    cases = load_cases()
    case_ids = _select_names(args.cases or ("E1" if args.smoke_test else ""), tuple(case["case_id"] for case in cases))
    selected_cases = [case for case in cases if case["case_id"] in case_ids]
    model_names = _select_names(args.models, LOGICAL_MODEL_ORDER)

    if args.dry_run:
        print(f"Dry run: {len(model_names)} models x {len(selected_cases)} cases = {len(model_names) * len(selected_cases)} calls")
        for name in model_names:
            for case in selected_cases:
                print(f"  {name} {case['case_id']}")
        print("No API calls were made.")
        return

    settings = load_settings()
    try:
        slots = _load_resolved(refresh=args.refresh_models)
    except ProviderError as exc:
        print(sanitize_text(str(exc)))
        if exc.auth:
            print('export OPENAI_API_KEY="..."')
            print('export OPENROUTER_API_KEY="..."')
            raise SystemExit(2)
        raise
    by_name = {slot["logical_name"]: slot for slot in slots}
    missing_slots = [name for name in model_names if name not in by_name]
    if missing_slots:
        raise SystemExit(f"Resolved model file is missing slots: {missing_slots}")
    selected_slots = [by_name[name] for name in model_names]

    run_id, created_at = _run_identity()
    done = set() if args.force else successful_pairs(RESPONSES_PATH)
    tasks = []
    for slot in selected_slots:
        for case in selected_cases:
            key = (slot["logical_name"], case["case_id"])
            if key in done:
                print(f"Skip {key[0]} {key[1]} (already successful)", flush=True)
                continue
            tasks.append((case, slot))

    workers = args.workers or int(settings.get("max_workers", 4))
    lock = threading.Lock()
    stop_auth = threading.Event()
    print(f"Executing {len(tasks)} calls with {workers} workers. Run id {run_id}", flush=True)

    def _worker(case: dict[str, Any], slot: dict[str, Any]) -> dict[str, Any]:
        if stop_auth.is_set():
            prompt = render_vlm_prompt(case["question"])
            record = _base_record(run_id, case, slot, prompt)
            record["error"] = "stopped after an authentication failure"
            return record
        record = execute_task(run_id, case, slot, settings)
        if record.get("status") != "success" and record.get("error") and "API key" in str(record.get("error")):
            stop_auth.set()
        return record

    completed = 0
    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(_worker, case, slot) for case, slot in tasks]
            for future in as_completed(futures):
                record = future.result()
                append_jsonl(RESPONSES_PATH, record, lock=lock)
                completed += 1
                print(
                    f"[{completed}/{len(tasks)}] {record['logical_model_name']} {record['case_id']} "
                    f"{record['status']} latency={record.get('latency_seconds')}",
                    flush=True,
                )

    write_manifest(run_id, created_at, cases, slots)
    if args.smoke_test:
        write_smoke_report(slots)
        print(f"Wrote {SMOKE_MD}", flush=True)

    latest = {}
    for row in read_jsonl(RESPONSES_PATH):
        latest[response_key(row)] = row
    successes = [row for row in latest.values() if row.get("status") == "success"]
    failures = [row for row in latest.values() if row.get("status") != "success"]
    print(f"Successful responses in {RESPONSES_PATH.name}: {len(successes)}")
    if failures:
        print(f"Failed responses: {len(failures)}")
        for row in failures:
            print(f"  {row.get('logical_model_name')} {row.get('case_id')}: {row.get('error')}")

if __name__ == "__main__":
    main()
