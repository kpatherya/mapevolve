from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from benchmark.src.models import load_model_config
from benchmark.src.paths import ARTIFACTS_DIR, JUDGE_RESULTS_PATH, RAW_API_DIR, RESPONSES_PATH
from benchmark.src.preflight import load_settings, require_env
from benchmark.src.prepare_data import load_cases
from benchmark.src.prompting import render_judge_prompt
from benchmark.src.providers.common import ProviderError
from benchmark.src.providers.openai_provider import generate_text
from benchmark.src.records import append_jsonl, latest_by, read_jsonl, response_key, utc_now
from benchmark.src.redact import sanitize_text
from benchmark.src.scoring import compute_totals, parse_judge_output

JUDGE_MODEL_PATH = ARTIFACTS_DIR / "judge_model.json"
REPAIR_INSTRUCTION = (
    "\n\nThe previous reply was not valid JSON matching the required schema. "
    "Return only one JSON object with the required keys and allowed score values. "
    "Do not include markdown or prose."
)

def anonymous_response_id(case_id: str, logical_name: str, response_text: str, seed: int) -> str:
    payload = f"{seed}|{case_id}|{logical_name}|{response_text}".encode("utf-8")
    return "R" + hashlib.sha256(payload).hexdigest()[:12]

def _judge_settings(settings: dict[str, Any]) -> dict[str, Any]:
    tuned = dict(settings)
    tuned["max_output_tokens"] = int(settings.get("judge_max_output_tokens", 1200))
    tuned["empty_output_retry_max_tokens"] = max(4000, tuned["max_output_tokens"])
    return tuned

def resolve_judge_model(settings: dict[str, Any], refresh: bool = False) -> dict[str, Any]:
    if JUDGE_MODEL_PATH.exists() and not refresh:
        return json.loads(JUDGE_MODEL_PATH.read_text(encoding="utf-8"))
    require_env(["OPENAI_API_KEY"])
    env_model = (os.environ.get("OPENAI_JUDGE_MODEL") or "").strip()
    fallbacks = list(load_model_config()["judge"]["fallback_model_ids"])
    candidates = [env_model] if env_model else fallbacks
    notes = []
    probe_settings = _judge_settings(settings)
    probe_settings["max_output_tokens"] = 64
    probe_settings["empty_output_retry_max_tokens"] = 256
    for model_id in candidates:
        try:
            result = generate_text(model_id, "Reply with the single word OK.", probe_settings)
        except ProviderError as exc:
            if exc.auth:
                print(sanitize_text(str(exc)))
                print('export OPENAI_API_KEY="..."')
                raise SystemExit(2)
            notes.append(f"{model_id}: {sanitize_text(str(exc))}")
            continue
        resolved = {
            "requested_model_id": candidates[0],
            "resolved_model_id": result.get("resolved_model_id") or model_id,
            "selection_notes": notes,
            "probe_reported_cost_usd": result.get("reported_cost_usd"),
            "probe_adjustments": result.get("actual_parameter_adjustments") or [],
        }
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        JUDGE_MODEL_PATH.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
        return resolved
    raise SystemExit("No OpenAI judge model accepted a text probe. " + " | ".join(notes))

def _successes() -> list[dict[str, Any]]:
    latest = latest_by(read_jsonl(RESPONSES_PATH), response_key)
    rows = [row for row in latest.values() if row.get("status") == "success"]
    rows.sort(key=lambda row: (row.get("logical_model_name") or "", row.get("case_id") or ""))
    return rows

def _already_judged() -> set[tuple[str, str, str]]:
    latest = latest_by(
        read_jsonl(JUDGE_RESULTS_PATH),
        lambda row: (row.get("logical_model_name"), row.get("case_id"), row.get("response_sha256")),
    )
    return {key for key, row in latest.items() if row.get("judge_status") == "success"}

def _score_text(text: str) -> Optional[dict[str, Any]]:
    parsed = parse_judge_output(text)
    if parsed is None:
        return None
    raw_total, final_total, label = compute_totals(
        parsed["core_change_score"],
        parsed["confounder_score"],
        parsed["hallucination_free_score"],
        parsed["hallucinated_change"],
    )
    parsed["raw_total"] = raw_total
    parsed["final_total"] = final_total
    parsed["label"] = label
    return parsed

def judge_one(
    response: dict[str, Any],
    case: dict[str, Any],
    judge_model: str,
    settings: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    prompt = render_judge_prompt(
        case["question"],
        case["ground_truth_raw"],
        case["rubric"],
        response.get("response_text") or "",
    )
    anon = anonymous_response_id(response["case_id"], response["logical_model_name"], response.get("response_text") or "", seed)
    record: dict[str, Any] = {
        "anonymous_response_id": anon,
        "case_id": response["case_id"],
        "logical_model_name": response["logical_model_name"],
        "resolved_model_id": response.get("resolved_model_id"),
        "judge_model": judge_model,
        "self_judged": response.get("resolved_model_id") == judge_model,
        "prompt_version": "judge_prompt_v1",
        "response_sha256": response.get("response_sha256"),
        "judge_status": "error",
        "saw_images": False,
        "candidate_identity_in_prompt": False,
        "repair_used": False,
        "timestamp_utc": utc_now(),
        "actual_parameter_adjustments": [],
        "input_tokens": None,
        "output_tokens": None,
        "reported_cost_usd_if_available": None,
        "error": None,
    }
    try:
        first = generate_text(judge_model, prompt, settings)
    except Exception as exc:
        record["error"] = sanitize_text(str(exc))
        _write_judge_sidecar(record, prompt, None, None)
        return record
    parsed = _score_text(first.get("response_text") or "")
    used = first
    raw_texts = [first.get("response_text") or ""]
    adjustments = list(first.get("actual_parameter_adjustments") or [])
    if parsed is None:
        repair_prompt = prompt + REPAIR_INSTRUCTION + "\n\nPrevious reply:\n" + (first.get("response_text") or "")
        try:
            second = generate_text(judge_model, repair_prompt, settings)
        except Exception as exc:
            record["error"] = "judge JSON was malformed and the repair call failed: " + sanitize_text(str(exc))
            record["repair_used"] = True
            record["actual_parameter_adjustments"] = adjustments
            _write_judge_sidecar(record, prompt, raw_texts, None)
            return record
        record["repair_used"] = True
        adjustments.extend(second.get("actual_parameter_adjustments") or [])
        adjustments.append("retried once with a JSON-repair instruction")
        raw_texts.append(second.get("response_text") or "")
        parsed = _score_text(second.get("response_text") or "")
        used = second
    if parsed is None:
        record["error"] = "judge output was not valid JSON after one repair attempt"
        record["actual_parameter_adjustments"] = adjustments
        record["input_tokens"] = _combine_usage(first.get("input_tokens"), used.get("input_tokens"), record["repair_used"])
        record["output_tokens"] = _combine_usage(first.get("output_tokens"), used.get("output_tokens"), record["repair_used"])
        record["reported_cost_usd_if_available"] = _combine_usage(
            first.get("reported_cost_usd"),
            used.get("reported_cost_usd"),
            record["repair_used"],
        )
        _write_judge_sidecar(record, prompt, raw_texts, used.get("raw_response"))
        return record
    if record["repair_used"]:
        input_tokens = _combine_usage(first.get("input_tokens"), used.get("input_tokens"), True)
        output_tokens = _combine_usage(first.get("output_tokens"), used.get("output_tokens"), True)
        cost = _combine_usage(first.get("reported_cost_usd"), used.get("reported_cost_usd"), True)
    else:
        input_tokens = first.get("input_tokens")
        output_tokens = first.get("output_tokens")
        cost = first.get("reported_cost_usd")
    record.update(parsed)
    record.update(
        {
            "judge_status": "success",
            "judge_reason": parsed["reason"],
            "actual_parameter_adjustments": adjustments,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reported_cost_usd_if_available": cost,
            "error": None,
            "timestamp_utc": utc_now(),
        }
    )
    _write_judge_sidecar(record, prompt, raw_texts, used.get("raw_response"))
    return record

def _combine_usage(left: Optional[float], right: Optional[float], repair: bool) -> Optional[float]:
    if not repair:
        return left
    if left is None or right is None:
        return None
    return left + right

def _write_judge_sidecar(record: dict[str, Any], prompt: str, raw_texts: Optional[list[str]], raw_response: Any) -> None:
    RAW_API_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_API_DIR / f"judge_{record['anonymous_response_id']}.json"
    payload = {
        "saw_images": False,
        "image_count": 0,
        "candidate_identity_in_prompt": False,
        "tools": None,
        "web_search": False,
        "prompt": prompt,
        "raw_judge_texts": raw_texts or [],
        "response": raw_response,
        "error": record.get("error"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Judge benchmark responses without showing images.")
    parser.add_argument("--resume", action="store_true", help="Skip responses that already have a successful judgment.")
    parser.add_argument("--force", action="store_true", help="Judge selected responses again.")
    parser.add_argument("--refresh-judge-model", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    settings = load_settings()
    judge = resolve_judge_model(settings, refresh=args.refresh_judge_model)
    judge_model = judge["resolved_model_id"]
    print(f"Judge model: {judge_model}", flush=True)
    cases = {case["case_id"]: case for case in load_cases()}
    seed = int(settings.get("judge_seed", 20261012))
    done = set() if args.force else _already_judged()
    pending = []
    for row in _successes():
        key = (row.get("logical_model_name"), row.get("case_id"), row.get("response_sha256"))
        if key in done:
            print(f"Skip judge {key[0]} {key[1]}", flush=True)
            continue
        pending.append(row)
    rng = random.Random(seed)
    rng.shuffle(pending)
    judge_settings = _judge_settings(settings)
    workers = args.workers or int(settings.get("max_workers", 4))
    lock = threading.Lock()
    print(f"Judging {len(pending)} responses.", flush=True)

    def _worker(row: dict[str, Any]) -> dict[str, Any]:
        return judge_one(row, cases[row["case_id"]], judge_model, judge_settings, seed)

    completed = 0
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(_worker, row) for row in pending]
            for future in as_completed(futures):
                record = future.result()
                append_jsonl(JUDGE_RESULTS_PATH, record, lock=lock)
                completed += 1
                print(
                    f"[{completed}/{len(pending)}] {record['logical_model_name']} {record['case_id']} "
                    f"{record['judge_status']} final={record.get('final_total')}",
                    flush=True,
                )
    judged = [
        row
        for row in latest_by(
            read_jsonl(JUDGE_RESULTS_PATH),
            lambda row: (row.get("logical_model_name"), row.get("case_id"), row.get("response_sha256")),
        ).values()
        if row.get("judge_status") == "success"
    ]
    print(f"Successful judgments: {len(judged)}")

if __name__ == "__main__":
    main()
