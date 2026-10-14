from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from benchmark.src.paths import (
    ARTIFACTS_DIR,
    AVAILABILITY_JSON,
    CASE_IDS,
    DIFFICULTIES,
    JOINED_RESULTS_PATH,
    JUDGE_RESULTS_PATH,
    LOGICAL_MODEL_ORDER,
    RESPONSES_PATH,
)
from benchmark.src.records import latest_by, read_jsonl, response_key
from benchmark.src.scoring import compute_totals

JUDGE_MODEL_PATH = ARTIFACTS_DIR / "judge_model.json"

def _cell(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))

def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

def _existing_human_notes(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    notes = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("anonymous_response_id") or ""
            if not key:
                continue
            notes[key] = {
                "human_score": row.get("human_score") or "",
                "human_label": row.get("human_label") or "",
                "human_hallucination": row.get("human_hallucination") or "",
                "human_notes": row.get("human_notes") or "",
            }
    return notes

def build_joined() -> list[dict[str, Any]]:
    responses = list(latest_by(read_jsonl(RESPONSES_PATH), response_key).values())
    judgments = latest_by(
        read_jsonl(JUDGE_RESULTS_PATH),
        lambda row: (row.get("logical_model_name"), row.get("case_id"), row.get("response_sha256")),
    )
    joined = []
    for response in responses:
        key = (response.get("logical_model_name"), response.get("case_id"), response.get("response_sha256"))
        judgment = judgments.get(key) if response.get("status") == "success" else None
        item = {
            "run_id": response.get("run_id"),
            "case_id": response.get("case_id"),
            "difficulty": response.get("difficulty"),
            "logical_model_name": response.get("logical_model_name"),
            "requested_model_id": response.get("requested_model_id"),
            "resolved_model_id": response.get("resolved_model_id"),
            "backend": response.get("backend"),
            "provider_if_available": response.get("provider_if_available"),
            "prompt_version": response.get("prompt_version"),
            "image_a_sha256": response.get("image_a_sha256"),
            "image_b_sha256": response.get("image_b_sha256"),
            "response_text": response.get("response_text") or "",
            "response_sha256": response.get("response_sha256"),
            "latency_seconds": response.get("latency_seconds"),
            "input_tokens": response.get("input_tokens"),
            "output_tokens": response.get("output_tokens"),
            "reasoning_tokens_if_reported": response.get("reasoning_tokens_if_reported"),
            "reported_cost_usd_if_available": response.get("reported_cost_usd_if_available"),
            "temperature_requested": response.get("temperature_requested"),
            "actual_parameter_adjustments": response.get("actual_parameter_adjustments") or [],
            "timestamp_utc": response.get("timestamp_utc"),
            "status": response.get("status"),
            "error": response.get("error"),
            "tools_enabled": response.get("tools_enabled"),
            "web_search": response.get("web_search"),
            "anonymous_response_id": None if judgment is None else judgment.get("anonymous_response_id"),
            "judge_status": None if judgment is None else judgment.get("judge_status"),
            "judge_model": None if judgment is None else judgment.get("judge_model"),
            "self_judged": None if judgment is None else judgment.get("self_judged"),
            "judge_error": None if judgment is None else judgment.get("error"),
            "judge_reported_cost_usd": None if judgment is None else judgment.get("reported_cost_usd_if_available"),
        }
        if judgment and judgment.get("judge_status") == "success":
            raw_total, final_total, label = compute_totals(
                int(judgment["core_change_score"]),
                int(judgment["confounder_score"]),
                int(judgment["hallucination_free_score"]),
                bool(judgment["hallucinated_change"]),
            )
            item.update(
                {
                    "core_change_score": judgment["core_change_score"],
                    "confounder_score": judgment["confounder_score"],
                    "hallucination_free_score": judgment["hallucination_free_score"],
                    "hallucinated_change": judgment["hallucinated_change"],
                    "raw_total": raw_total,
                    "final_total": final_total,
                    "label": label,
                    "supported_points": judgment.get("supported_points") or [],
                    "missing_points": judgment.get("missing_points") or [],
                    "unsupported_points": judgment.get("unsupported_points") or [],
                    "judge_reason": judgment.get("judge_reason") or judgment.get("reason"),
                }
            )
        joined.append(item)
    order = {name: index for index, name in enumerate(LOGICAL_MODEL_ORDER)}
    case_order = {case_id: index for index, case_id in enumerate(CASE_IDS)}
    joined.sort(key=lambda row: (order.get(row["logical_model_name"], 99), case_order.get(row["case_id"], 99)))
    return joined

def _scored(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("final_total") is not None]

def _summary_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = _scored(rows)
    return {
        "n_cases": len(scored),
        "mean_final_score": _mean([float(row["final_total"]) for row in scored]),
        "correct_count": sum(1 for row in scored if row["label"] == "correct"),
        "partial_count": sum(1 for row in scored if row["label"] == "partially_correct"),
        "incorrect_count": sum(1 for row in scored if row["label"] == "incorrect"),
        "hallucination_count": sum(1 for row in scored if row.get("hallucinated_change") is True),
    }

def write_tables(joined: list[dict[str, Any]]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    response_columns = [
        "run_id",
        "case_id",
        "difficulty",
        "logical_model_name",
        "requested_model_id",
        "resolved_model_id",
        "backend",
        "provider_if_available",
        "prompt_version",
        "image_a_sha256",
        "image_b_sha256",
        "response_text",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens_if_reported",
        "reported_cost_usd_if_available",
        "temperature_requested",
        "timestamp_utc",
        "status",
        "error",
    ]
    _write_csv(ARTIFACTS_DIR / "all_responses.csv", response_columns, joined)

    judged_columns = [
        "case_id",
        "difficulty",
        "anonymous_response_id",
        "logical_model_name",
        "resolved_model_id",
        "judge_model",
        "self_judged",
        "response_text",
        "core_change_score",
        "confounder_score",
        "hallucination_free_score",
        "hallucinated_change",
        "raw_total",
        "final_total",
        "label",
        "judge_reason",
        "supported_points",
        "missing_points",
        "unsupported_points",
        "latency_seconds",
        "reported_cost_usd_if_available",
    ]
    judged_rows = []
    for row in _scored(joined):
        exported = dict(row)
        for key in ("supported_points", "missing_points", "unsupported_points"):
            exported[key] = " | ".join(row.get(key) or [])
        judged_rows.append(exported)
    _write_csv(ARTIFACTS_DIR / "judged_responses.csv", judged_columns, judged_rows)

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_model[row["logical_model_name"]].append(row)

    model_rows = []
    difficulty_rows = []
    hallucination_rows = []
    cost_rows = []
    for name in LOGICAL_MODEL_ORDER:
        rows = by_model.get(name, [])
        if not rows:
            continue
        counts = _summary_counts(rows)
        successes = [row for row in rows if row.get("status") == "success"]
        costs = [float(row["reported_cost_usd_if_available"]) for row in successes if row.get("reported_cost_usd_if_available") is not None]
        latencies = [float(row["latency_seconds"]) for row in successes if row.get("latency_seconds") is not None]
        model_rows.append(
            {
                "logical_model_name": name,
                "resolved_model_id": rows[0].get("resolved_model_id") or "",
                "n_cases": counts["n_cases"],
                "mean_final_score": _cell(counts["mean_final_score"]),
                "correct_count": counts["correct_count"],
                "partial_count": counts["partial_count"],
                "incorrect_count": counts["incorrect_count"],
                "hallucination_count": counts["hallucination_count"],
                "hallucination_rate": _cell(
                    None if counts["n_cases"] == 0 else counts["hallucination_count"] / float(counts["n_cases"])
                ),
                "mean_latency_seconds": _cell(_mean(latencies)),
                "total_reported_cost_usd": _cell(None if not costs else sum(costs)),
            }
        )
        for difficulty in DIFFICULTIES:
            subset = [row for row in rows if row.get("difficulty") == difficulty]
            subset_counts = _summary_counts(subset)
            difficulty_rows.append(
                {
                    "logical_model_name": name,
                    "resolved_model_id": rows[0].get("resolved_model_id") or "",
                    "difficulty": difficulty,
                    "n_cases": subset_counts["n_cases"],
                    "mean_final_score": _cell(subset_counts["mean_final_score"]),
                    "correct_count": subset_counts["correct_count"],
                    "partial_count": subset_counts["partial_count"],
                    "incorrect_count": subset_counts["incorrect_count"],
                    "hallucination_count": subset_counts["hallucination_count"],
                }
            )
            hallucination_rows.append(
                {
                    "logical_model_name": name,
                    "resolved_model_id": rows[0].get("resolved_model_id") or "",
                    "difficulty": difficulty,
                    "n_cases": subset_counts["n_cases"],
                    "hallucination_count": subset_counts["hallucination_count"],
                    "hallucination_rate": _cell(
                        None
                        if subset_counts["n_cases"] == 0
                        else subset_counts["hallucination_count"] / float(subset_counts["n_cases"])
                    ),
                }
            )
        hallucination_rows.append(
            {
                "logical_model_name": name,
                "resolved_model_id": rows[0].get("resolved_model_id") or "",
                "difficulty": "all",
                "n_cases": counts["n_cases"],
                "hallucination_count": counts["hallucination_count"],
                "hallucination_rate": _cell(
                    None if counts["n_cases"] == 0 else counts["hallucination_count"] / float(counts["n_cases"])
                ),
            }
        )
        input_tokens = [int(row["input_tokens"]) for row in successes if row.get("input_tokens") is not None]
        output_tokens = [int(row["output_tokens"]) for row in successes if row.get("output_tokens") is not None]
        cost_rows.append(
            {
                "logical_model_name": name,
                "resolved_model_id": rows[0].get("resolved_model_id") or "",
                "n_success": len(successes),
                "n_error": sum(1 for row in rows if row.get("status") != "success"),
                "mean_latency_seconds": _cell(_mean(latencies)),
                "total_reported_cost_usd": _cell(None if not costs else sum(costs)),
                "n_with_reported_cost": len(costs),
                "total_input_tokens": "" if not input_tokens else sum(input_tokens),
                "total_output_tokens": "" if not output_tokens else sum(output_tokens),
                "n_with_input_tokens": len(input_tokens),
                "n_with_output_tokens": len(output_tokens),
            }
        )

    for difficulty in DIFFICULTIES:
        subset = [row for row in joined if row.get("difficulty") == difficulty]
        counts = _summary_counts(subset)
        difficulty_rows.append(
            {
                "logical_model_name": "ALL",
                "resolved_model_id": "",
                "difficulty": difficulty,
                "n_cases": counts["n_cases"],
                "mean_final_score": _cell(counts["mean_final_score"]),
                "correct_count": counts["correct_count"],
                "partial_count": counts["partial_count"],
                "incorrect_count": counts["incorrect_count"],
                "hallucination_count": counts["hallucination_count"],
            }
        )

    case_rows = []
    for case_id in CASE_IDS:
        subset = [row for row in joined if row.get("case_id") == case_id]
        counts = _summary_counts(subset)
        case_rows.append(
            {
                "case_id": case_id,
                "difficulty": subset[0]["difficulty"] if subset else "",
                "n_models": counts["n_cases"],
                "mean_final_score": _cell(counts["mean_final_score"]),
                "correct_count": counts["correct_count"],
                "partial_count": counts["partial_count"],
                "incorrect_count": counts["incorrect_count"],
                "hallucination_count": counts["hallucination_count"],
            }
        )

    matrix_rows = []
    for name in LOGICAL_MODEL_ORDER:
        rows = by_model.get(name, [])
        if not rows:
            continue
        matrix = {
            "logical_model_name": name,
            "resolved_model_id": rows[0].get("resolved_model_id") or "",
        }
        by_case = {row["case_id"]: row for row in rows}
        for case_id in CASE_IDS:
            row = by_case.get(case_id)
            matrix[case_id] = "" if row is None or row.get("final_total") is None else row["final_total"]
        matrix_rows.append(matrix)

    _write_csv(
        ARTIFACTS_DIR / "model_summary.csv",
        [
            "logical_model_name",
            "resolved_model_id",
            "n_cases",
            "mean_final_score",
            "correct_count",
            "partial_count",
            "incorrect_count",
            "hallucination_count",
            "hallucination_rate",
            "mean_latency_seconds",
            "total_reported_cost_usd",
        ],
        model_rows,
    )
    _write_csv(
        ARTIFACTS_DIR / "difficulty_summary.csv",
        [
            "logical_model_name",
            "resolved_model_id",
            "difficulty",
            "n_cases",
            "mean_final_score",
            "correct_count",
            "partial_count",
            "incorrect_count",
            "hallucination_count",
        ],
        difficulty_rows,
    )
    _write_csv(
        ARTIFACTS_DIR / "case_summary.csv",
        [
            "case_id",
            "difficulty",
            "n_models",
            "mean_final_score",
            "correct_count",
            "partial_count",
            "incorrect_count",
            "hallucination_count",
        ],
        case_rows,
    )
    _write_csv(
        ARTIFACTS_DIR / "model_case_matrix.csv",
        ["logical_model_name", "resolved_model_id", *CASE_IDS],
        matrix_rows,
    )
    _write_csv(
        ARTIFACTS_DIR / "hallucination_summary.csv",
        [
            "logical_model_name",
            "resolved_model_id",
            "difficulty",
            "n_cases",
            "hallucination_count",
            "hallucination_rate",
        ],
        hallucination_rows,
    )
    smoke_cost, smoke_n = _smoke_cost()
    judge_cost, judge_cost_n = _judge_cost(joined)
    judge_n = sum(1 for row in joined if row.get("judge_status") == "success")
    cost_rows.append(
        {
            "logical_model_name": "SMOKE_PREFLIGHT",
            "resolved_model_id": "",
            "n_success": smoke_n,
            "n_error": "",
            "mean_latency_seconds": "",
            "total_reported_cost_usd": _cell(smoke_cost),
            "n_with_reported_cost": smoke_n if smoke_cost is not None else 0,
            "total_input_tokens": "",
            "total_output_tokens": "",
            "n_with_input_tokens": "",
            "n_with_output_tokens": "",
        }
    )
    cost_rows.append(
        {
            "logical_model_name": "JUDGE",
            "resolved_model_id": _judge_model_id(),
            "n_success": judge_n,
            "n_error": "",
            "mean_latency_seconds": "",
            "total_reported_cost_usd": _cell(judge_cost),
            "n_with_reported_cost": judge_cost_n if judge_cost is not None else 0,
            "total_input_tokens": "",
            "total_output_tokens": "",
            "n_with_input_tokens": "",
            "n_with_output_tokens": "",
        }
    )
    _write_csv(
        ARTIFACTS_DIR / "cost_latency_summary.csv",
        [
            "logical_model_name",
            "resolved_model_id",
            "n_success",
            "n_error",
            "mean_latency_seconds",
            "total_reported_cost_usd",
            "n_with_reported_cost",
            "total_input_tokens",
            "total_output_tokens",
            "n_with_input_tokens",
            "n_with_output_tokens",
        ],
        cost_rows,
    )
    _write_manual_review(joined)

def _judge_model_id() -> str:
    if not JUDGE_MODEL_PATH.exists():
        return ""
    return json.loads(JUDGE_MODEL_PATH.read_text(encoding="utf-8")).get("resolved_model_id") or ""

def _smoke_cost() -> tuple[Optional[float], int]:
    if not AVAILABILITY_JSON.exists():
        return None, 0
    rows = json.loads(AVAILABILITY_JSON.read_text(encoding="utf-8"))
    values = [float(row["smoke_reported_cost_usd"]) for row in rows if row.get("smoke_reported_cost_usd") is not None]
    if not values:
        return None, 0
    return sum(values), len(values)

def _judge_cost(joined: list[dict[str, Any]]) -> tuple[Optional[float], int]:
    values = [
        float(row["judge_reported_cost_usd"])
        for row in joined
        if row.get("judge_reported_cost_usd") is not None
    ]
    probe = None
    if JUDGE_MODEL_PATH.exists():
        probe = json.loads(JUDGE_MODEL_PATH.read_text(encoding="utf-8")).get("probe_reported_cost_usd")
    if probe is not None:
        values.append(float(probe))
    if not values:
        return None, 0
    return sum(values), len(values)

def _write_manual_review(joined: list[dict[str, Any]]) -> None:
    path = ARTIFACTS_DIR / "manual_review.csv"
    existing = _existing_human_notes(path)
    columns = [
        "case_id",
        "anonymous_response_id",
        "model",
        "model_response",
        "automatic_score",
        "automatic_label",
        "hallucinated_change",
        "judge_reason",
        "human_score",
        "human_label",
        "human_hallucination",
        "human_notes",
    ]
    rows = []
    for row in joined:
        if row.get("status") != "success":
            continue
        anon = row.get("anonymous_response_id") or ""
        preserved = existing.get(anon, {})
        rows.append(
            {
                "case_id": row["case_id"],
                "anonymous_response_id": anon,
                "model": row["logical_model_name"],
                "model_response": row.get("response_text") or "",
                "automatic_score": "" if row.get("final_total") is None else row["final_total"],
                "automatic_label": row.get("label") or "",
                "hallucinated_change": "" if row.get("hallucinated_change") is None else row["hallucinated_change"],
                "judge_reason": row.get("judge_reason") or "",
                "human_score": preserved.get("human_score", ""),
                "human_label": preserved.get("human_label", ""),
                "human_hallucination": preserved.get("human_hallucination", ""),
                "human_notes": preserved.get("human_notes", ""),
            }
        )
    _write_csv(path, columns, rows)

def main() -> None:
    joined = build_joined()
    JOINED_RESULTS_PATH.write_text(json.dumps(joined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_tables(joined)
    scored = _scored(joined)
    print(f"Wrote tables for {len(joined)} responses, {len(scored)} scored.")

if __name__ == "__main__":
    main()
