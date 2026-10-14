from __future__ import annotations

import json

import pytest

from benchmark.src.providers.common import ProviderError, call_with_policy, needs_output_room
from benchmark.src.scoring import (
    compute_totals,
    failure_interpretation,
    parse_judge_output,
    select_representative_failures,
)

def _valid(**overrides):
    payload = {
        "core_change_score": 2,
        "confounder_score": 1,
        "hallucination_free_score": 1,
        "hallucinated_change": False,
        "supported_points": ["no structural change"],
        "missing_points": [],
        "unsupported_points": [],
        "reason": "The response matches the answer key.",
    }
    payload.update(overrides)
    return payload

def test_deterministic_scores_and_hallucination_cap() -> None:
    assert compute_totals(2, 1, 1, False) == (4, 4, "correct")
    assert compute_totals(2, 1, 1, True) == (4, 3, "partially_correct")
    assert compute_totals(1, 1, 1, True) == (3, 3, "partially_correct")
    assert compute_totals(2, 0, 0, False) == (2, 2, "partially_correct")
    assert compute_totals(1, 0, 0, True) == (1, 1, "incorrect")
    assert compute_totals(0, 0, 0, False) == (0, 0, "incorrect")

def test_malformed_judge_json_is_not_inferred() -> None:
    assert parse_judge_output("The core score is 2 and there is no hallucination.") is None
    assert parse_judge_output("") is None
    assert parse_judge_output("{not json}") is None
    parsed = parse_judge_output(json.dumps(_valid()))
    assert parsed is not None
    assert parsed["core_change_score"] == 2
    fenced = "```json\n" + json.dumps(_valid(core_change_score=1)) + "\n```"
    assert parse_judge_output(fenced)["core_change_score"] == 1
    assert parse_judge_output(json.dumps(_valid(core_change_score=5))) is None
    assert parse_judge_output(json.dumps(_valid(confounder_score=True))) is None
    assert parse_judge_output(json.dumps(_valid(hallucinated_change="false"))) is None
    assert parse_judge_output(json.dumps(_valid(reason="  "))) is None
    missing = _valid()
    del missing["missing_points"]
    assert parse_judge_output(json.dumps(missing)) is None
    assert parse_judge_output(json.dumps(_valid(supported_points=["ok", 1]))) is None

def test_failure_selection_is_reproducible_and_diverse() -> None:
    rows = [
        {"uid": "a", "case_id": "M1", "difficulty": "medium", "model": "GEMINI", "final_total": 1, "hallucinated_change": True},
        {"uid": "b", "case_id": "M1", "difficulty": "medium", "model": "CLAUDE", "final_total": 0, "hallucinated_change": True},
        {"uid": "c", "case_id": "H1", "difficulty": "hard", "model": "QWEN", "final_total": 0, "hallucinated_change": True},
        {"uid": "d", "case_id": "H2", "difficulty": "hard", "model": "LLAMA", "final_total": 1, "hallucinated_change": False},
        {"uid": "e", "case_id": "E1", "difficulty": "easy", "model": "GEMMA", "final_total": 0, "hallucinated_change": False},
    ]
    selected = select_representative_failures(rows, k=4)
    assert [row["uid"] for row in selected] == [row["uid"] for row in select_representative_failures(rows, k=4)]
    assert {row["uid"] for row in selected[:3]} == {"b", "c", "a"} or selected[0]["hallucinated_change"] is True
    models = [row["model"] for row in selected]
    assert len(models) == len(set(models))
    assert failure_interpretation(
        {"hallucinated_change": True, "unsupported_points": ["The roof was replaced"], "missing_points": []}
    ).startswith("The response asserts")

class _Boom(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status

def test_retry_strips_temperature_and_does_not_loop_on_modality() -> None:
    calls = []

    def request(params):
        calls.append(dict(params))
        if "temperature" in params:
            raise _Boom(400, "Unsupported parameter: temperature is not supported")
        return "ok"

    result, adjustments = call_with_policy(
        request,
        {"temperature": 0, "max_output_tokens": 350},
        {"max_attempts": 5, "retry_backoff_seconds": [1, 2, 4, 8]},
        sleep_fn=lambda _seconds: None,
    )
    assert result == "ok"
    assert "temperature" not in calls[-1]
    assert adjustments

    modality_calls = {"n": 0}

    def reject(_params):
        modality_calls["n"] += 1
        raise _Boom(400, "This model does not support image input")

    with pytest.raises(ProviderError) as caught:
        call_with_policy(
            reject,
            {"temperature": 0},
            {"max_attempts": 5, "retry_backoff_seconds": [1, 2, 4, 8]},
            sleep_fn=lambda _seconds: None,
        )
    assert caught.value.modality is True
    assert modality_calls["n"] == 1

    transient = {"n": 0}

    def flaky(_params):
        transient["n"] += 1
        if transient["n"] < 3:
            raise _Boom(503, "unavailable")
        return "recovered"

    recovered, _notes = call_with_policy(
        flaky,
        {},
        {"max_attempts": 5, "retry_backoff_seconds": [1, 2, 4, 8]},
        sleep_fn=lambda _seconds: None,
    )
    assert recovered == "recovered"
    assert transient["n"] == 3

def test_truncated_output_needs_a_larger_budget() -> None:
    assert needs_output_room({"response_text": "", "finish_reason": "stop"}) is True
    assert needs_output_room({"response_text": "cut off", "finish_reason": "length"}) is True
    assert needs_output_room({"response_text": "No structural change is visible.", "finish_reason": "stop"}) is False
