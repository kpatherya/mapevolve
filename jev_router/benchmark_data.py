"""Load the pilot benchmark artifacts and turn them into per-model metrics.

Scoring source: `benchmark/artifacts/judged_responses.csv` holds GPT-5.1
LLM-judge scores (0-4). `benchmark/artifacts/manual_review.csv` has columns for
human adjudication; when a `human_score` is filled in it overrides the judge
score for that response. At the time of writing every human column is empty,
so all metrics are LLM-judge measurements.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

from jev_router.schemas import normalize_difficulty

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
CASES_PATH = BENCH / "cases.json"
JUDGED_PATH = BENCH / "artifacts" / "judged_responses.csv"
MANUAL_PATH = BENCH / "artifacts" / "manual_review.csv"
RAW_PATH = BENCH / "runs" / "raw_responses.jsonl"

# Every pilot case is a two-image "what changed?" question about one location.
# This label is assigned from the benchmark design, never from answer keys.
PILOT_TASK_TYPE = "temporal_comparison"
MAX_SCORE = 4.0


@dataclass
class ResponseRecord:
    case_id: str
    difficulty: str                 # normalized: easy / moderate / hard
    task_type: str
    model_key: str
    model_id: str
    score: float                    # 0..4
    label: str                      # correct / partial / incorrect
    unsupported_claim: bool
    score_source: str               # "llm_judge" | "human"
    latency_seconds: Optional[float]
    reported_cost_usd: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    response_text: str


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _float(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_tokens(path: Path = RAW_PATH) -> dict[tuple[str, str], dict[str, Any]]:
    """Last successful raw row per (case, model); retries appear earlier in the file."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") == "success":
            out[(row["case_id"], row["logical_model_name"])] = row
    return out


def _human_overrides(path: Path = MANUAL_PATH) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {
            row["anonymous_response_id"]: row
            for row in csv.DictReader(fh)
            if row.get("human_score", "").strip()
        }


def load_responses(
    judged_path: Path = JUDGED_PATH,
    manual_path: Path = MANUAL_PATH,
    raw_path: Path = RAW_PATH,
) -> list[ResponseRecord]:
    raw = _raw_tokens(raw_path)
    human = _human_overrides(manual_path)
    records = []
    with judged_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row["logical_model_name"]
            override = human.get(row["anonymous_response_id"])
            if override:
                score = float(override["human_score"])
                label = override.get("human_label") or row["label"]
                unsupported = _bool(override.get("human_hallucination", row["hallucinated_change"]))
                source = "human"
            else:
                score = float(row["final_total"])
                label = row["label"]
                unsupported = _bool(row["hallucinated_change"])
                source = "llm_judge"
            raw_row = raw.get((row["case_id"], key), {})
            records.append(
                ResponseRecord(
                    case_id=row["case_id"],
                    difficulty=normalize_difficulty(row["difficulty"]),
                    task_type=PILOT_TASK_TYPE,
                    model_key=key,
                    model_id=row["resolved_model_id"],
                    score=score,
                    label=normalize_label(label),
                    unsupported_claim=unsupported,
                    score_source=source,
                    latency_seconds=_float(row["latency_seconds"]),
                    reported_cost_usd=_float(row["reported_cost_usd_if_available"]),
                    input_tokens=raw_row.get("input_tokens"),
                    output_tokens=raw_row.get("output_tokens"),
                    response_text=row["response_text"],
                )
            )
    return records


def normalize_label(label: str) -> str:
    label = label.strip().lower()
    return "partial" if label in {"partial", "partially_correct", "partially correct"} else label


def _summary(rows: list[ResponseRecord]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "mean_score": round(mean(r.score for r in rows), 4),
        "normalized": round(mean(r.score for r in rows) / MAX_SCORE, 4),
        "correct": sum(r.label == "correct" for r in rows),
        "partial": sum(r.label == "partial" for r in rows),
        "incorrect": sum(r.label == "incorrect" for r in rows),
        "unsupported_claim_rate": round(mean(r.unsupported_claim for r in rows), 4),
    }


def model_metrics(
    records: Iterable[ResponseRecord], exclude_cases: Iterable[str] = ()
) -> dict[str, dict[str, Any]]:
    """Per-model general, task and task x difficulty metrics.

    `exclude_cases` supports leave-one-out evaluation so a case's own outcome is
    never used to route that case.
    """
    excluded = set(exclude_cases)
    by_model: dict[str, list[ResponseRecord]] = defaultdict(list)
    for rec in records:
        if rec.case_id not in excluded:
            by_model[rec.model_key].append(rec)

    out: dict[str, dict[str, Any]] = {}
    for key, rows in by_model.items():
        tasks: dict[str, Any] = {}
        for task in sorted({r.task_type for r in rows}):
            task_rows = [r for r in rows if r.task_type == task]
            by_diff = {
                d: _summary([r for r in task_rows if r.difficulty == d])
                for d in sorted({r.difficulty for r in task_rows})
            }
            tasks[task] = {"overall": _summary(task_rows), "by_difficulty": by_diff}
        lat = [r.latency_seconds for r in rows if r.latency_seconds is not None]
        cost = [r.reported_cost_usd for r in rows if r.reported_cost_usd is not None]
        tin = [r.input_tokens for r in rows if r.input_tokens is not None]
        tout = [r.output_tokens for r in rows if r.output_tokens is not None]
        out[key] = {
            "general": _summary(rows),
            "task_scores": tasks,
            "latency": {"mean_seconds": round(mean(lat), 3) if lat else None, "n": len(lat)},
            "observed_cost": {
                "mean_usd_per_query": round(mean(cost), 6) if cost else None,
                "n": len(cost),
            },
            "tokens": {
                "mean_input_tokens_per_query": round(mean(tin)) if tin else None,
                "mean_output_tokens_per_query": round(mean(tout)) if tout else None,
                "images_per_query": 2,
                "n": len(tin),
            },
            "score_sources": sorted({r.score_source for r in rows}),
        }
    return out
