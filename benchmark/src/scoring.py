from __future__ import annotations

import json
import re
from typing import Any, Optional

REQUIRED_JUDGE_KEYS = (
    "core_change_score",
    "confounder_score",
    "hallucination_free_score",
    "hallucinated_change",
    "supported_points",
    "missing_points",
    "unsupported_points",
    "reason",
)

DIFFICULTY_RANK = {"hard": 0, "medium": 1, "easy": 2}

def compute_totals(
    core_change_score: int,
    confounder_score: int,
    hallucination_free_score: int,
    hallucinated_change: bool,
) -> tuple[int, int, str]:
    raw_total = core_change_score + confounder_score + hallucination_free_score
    if hallucinated_change:
        final_total = min(raw_total, 3)
    else:
        final_total = raw_total
    if final_total == 4:
        label = "correct"
    elif final_total in (2, 3):
        label = "partially_correct"
    elif final_total in (0, 1):
        label = "incorrect"
    else:
        raise ValueError(f"final_total {final_total} is outside 0-4")
    return raw_total, final_total, label

def _strip_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    return cleaned.strip()

def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    if text is None:
        return None
    cleaned = _strip_fence(text)
    if not cleaned:
        return None
    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None

def validate_judge_payload(obj: Any) -> Optional[dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    for key in REQUIRED_JUDGE_KEYS:
        if key not in obj:
            return None
    core = obj["core_change_score"]
    confounder = obj["confounder_score"]
    free = obj["hallucination_free_score"]

    if type(core) is not int or core not in (0, 1, 2):
        return None
    if type(confounder) is not int or confounder not in (0, 1):
        return None
    if type(free) is not int or free not in (0, 1):
        return None
    if type(obj["hallucinated_change"]) is not bool:
        return None
    normalized_lists: dict[str, list[str]] = {}
    for key in ("supported_points", "missing_points", "unsupported_points"):
        value = obj[key]
        if not isinstance(value, list) or any(type(item) is not str for item in value):
            return None
        normalized_lists[key] = list(value)
    reason = obj["reason"]
    if type(reason) is not str or not reason.strip():
        return None
    return {
        "core_change_score": core,
        "confounder_score": confounder,
        "hallucination_free_score": free,
        "hallucinated_change": obj["hallucinated_change"],
        "supported_points": normalized_lists["supported_points"],
        "missing_points": normalized_lists["missing_points"],
        "unsupported_points": normalized_lists["unsupported_points"],
        "reason": reason.strip(),
    }

def parse_judge_output(text: str) -> Optional[dict[str, Any]]:
    obj = extract_json_object(text)
    if obj is None:
        return None
    return validate_judge_payload(obj)

def select_representative_failures(rows: list[dict[str, Any]], k: int = 4) -> list[dict[str, Any]]:
    usable = [row for row in rows if row.get("final_total") is not None and row.get("uid")]
    pools = [
        [
            row
            for row in usable
            if row.get("hallucinated_change") is True and row.get("difficulty") in ("medium", "hard")
        ],
        [row for row in usable if row.get("difficulty") == "hard"],
        [row for row in usable if row.get("difficulty") == "medium"],
        list(usable),
    ]

    def sort_key(row: dict[str, Any]) -> tuple:
        return (
            int(row["final_total"]),
            DIFFICULTY_RANK.get(str(row.get("difficulty")), 9),
            str(row.get("case_id")),
            str(row.get("model")),
            str(row.get("uid")),
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for pool in pools:
        ordered = sorted(pool, key=sort_key)
        for required_novelty in (2, 1, 0):
            for row in ordered:
                if len(selected) >= k:
                    return selected
                uid = str(row["uid"])
                if uid in selected_ids:
                    continue
                models = {str(item["model"]) for item in selected}
                cases = {str(item["case_id"]) for item in selected}
                novelty = int(str(row["model"]) not in models) + int(str(row["case_id"]) not in cases)
                if novelty >= required_novelty:
                    selected.append(row)
                    selected_ids.add(uid)
    return selected

def failure_interpretation(row: dict[str, Any]) -> str:
    if row.get("hallucinated_change") is True:
        unsupported = row.get("unsupported_points") or []
        if unsupported:
            detail = unsupported[0].rstrip(".")
            return (
                "The response asserts a physical or structural change that the answer key "
                f"does not support: {detail}."
            )
        return (
            "The response asserts a physical or structural change that the answer key does not support."
        )
    missing = row.get("missing_points") or []
    if missing:
        detail = missing[0].rstrip(".")
        return f"The response misses an answer-key point: {detail}."
    reason = str(row.get("judge_reason") or "").strip()
    if reason:
        return reason if reason.endswith(".") else reason + "."
    return "The response only partially matches the answer key."
