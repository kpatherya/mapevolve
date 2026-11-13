from __future__ import annotations

import csv
import json
import sys
from types import SimpleNamespace

import pytest

from jev_router.dataset import load_all
from jev_router.graded import Graded, append_graded, load_graded


# ---------------------------------------------------------------- review flag
def test_review_flag_requires_reviewer():
    from scripts.build_demo_manifest import _labels

    assert _labels({"id": "x"}) == {"author": "draft_by_claude", "reviewed": False,
                                     "note": "Drafted after viewing the images; needs human review."}
    out = _labels({"id": "x", "reviewed": True, "reviewer": "Ada", "review_note": "fixed difficulty"})
    assert out == {"author": "draft_by_claude", "reviewed": True, "reviewer": "Ada", "review_note": "fixed difficulty"}
    with pytest.raises(SystemExit):
        _labels({"id": "x", "reviewed": True})


# ---------------------------------------------------------------- stratified sample
def test_stratified_sample_covers_every_stratum_and_model():
    from scripts.grade_responses import stratified_sample

    cases = {c.case_id: c for c in load_all() if c.dataset == "demo_world"}
    rows = [{"case_id": cid, "model_key": m, "status": "ok", "answer": "x"} for cid in cases for m in ("A", "B", "C", "D")]
    picked = stratified_sample(rows, cases, 0.2, seed=1)
    strata = {(c.query_type, c.difficulty) for c in cases.values()}
    assert {(cases[r["case_id"]].query_type, cases[r["case_id"]].difficulty) for r in picked} == strata
    assert {r["model_key"] for r in picked} == {"A", "B", "C", "D"}
    assert 0.18 * len(rows) <= len(picked) <= 0.35 * len(rows)
    assert picked == stratified_sample(rows, cases, 0.2, seed=1)  # deterministic


# ---------------------------------------------------------------- agreement
def test_kappas():
    from scripts.judge_agreement import cohen_kappa, weighted_kappa

    assert weighted_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == 1.0
    assert weighted_kappa([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]) == pytest.approx(-1.0)
    assert cohen_kappa([True, False, True, False], [True, False, False, False]) == pytest.approx(0.5)


def test_agreement_end_to_end(tmp_path, monkeypatch):
    from scripts import judge_agreement

    graded = tmp_path / "graded"
    monkeypatch.setattr(judge_agreement, "GRADED_DIR", graded)
    monkeypatch.setattr(judge_agreement, "OUT", tmp_path / "grading")
    cid = next(c.case_id for c in load_all())
    judge = [Graded(cid, m, s, "x", u, "llm_judge:gpt-5.1", run_id="r1") for m, s, u in
             [("A", 4, False), ("B", 3, True), ("C", 1, True), ("D", 2, False)]]
    human = [Graded(cid, m, s, "x", u, "human", run_id="r1", grader="Ada") for m, s, u in
             [("A", 4, False), ("B", 2, True), ("C", 1, True), ("D", 3, False)]]
    append_graded(graded / "run.judge.jsonl", judge)
    append_graded(graded / "run.human.jsonl", human)
    assert judge_agreement.main() == 0
    rep = json.loads((tmp_path / "grading" / "agreement.json").read_text())
    assert rep["overall"]["n"] == 4 and rep["overall"]["exact"] == 0.5 and rep["overall"]["within_one"] == 1.0
    assert rep["overall"]["unsupported_kappa"] == 1.0


def test_human_import_validates_scores(tmp_path, monkeypatch):
    from scripts import grade_responses

    monkeypatch.setattr(grade_responses, "GRADED_DIR", tmp_path / "graded")
    monkeypatch.setattr(grade_responses, "ROOT", tmp_path)
    sheet = tmp_path / "sheet.csv"
    with sheet.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "model_key", "run_id", "model_answer", "latency_seconds", "cost_usd",
                    "human_score", "human_label", "human_unsupported_claim", "grader"])
        w.writerow(["GT-E1", "A", "r1", "ok", "", "", "3", "", "yes", "Ada"])
        w.writerow(["GT-E1", "B", "r1", "ok", "", "", "7", "", "", "Ada"])     # out of range
        w.writerow(["GT-E1", "C", "r1", "ok", "", "", "good", "", "", "Ada"])  # not a number
        w.writerow(["GT-E1", "D", "r1", "ok", "", "", "", "", "", ""])         # not graded yet
    assert grade_responses.import_human(SimpleNamespace(path=str(sheet))) == 0
    rows = [json.loads(l) for l in (tmp_path / "graded" / "sheet.human.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["score"] == 3 and rows[0]["unsupported"] and rows[0]["grader"] == "Ada"


def test_keep_repeats(tmp_path, monkeypatch):
    monkeypatch.setattr("jev_router.graded.GRADED_DIR", tmp_path)
    append_graded(tmp_path / "x.judge.jsonl", [Graded("GT-E1", "A", 4, "c", False, "llm_judge:x", run_id="r1"),
                                               Graded("GT-E1", "A", 2, "p", False, "llm_judge:x", run_id="r2")])
    live = lambda **k: [g for g in load_graded(**k) if g.score_source == "llm_judge:x"]
    assert len(live(keep_repeats=True)) == 2 and len(live()) == 1


# ---------------------------------------------------------------- answer check eval
def test_auc_and_threshold_helpers():
    from scripts.evaluate_answer_check import at_threshold, auc

    assert auc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert auc([0.5], [True]) is None
    t = at_threshold([0.9, 0.6, 0.3], [True, False, True], 0.5)
    assert t["flagged"] == 2 and t["precision"] == 0.5 and t["recall"] == 0.5


def test_answer_check_eval_runs_on_pilot(tmp_path, monkeypatch):
    from scripts import evaluate_answer_check

    monkeypatch.setattr("jev_router.classify_cache.CACHE", tmp_path / "cache")
    monkeypatch.setattr(evaluate_answer_check, "OUT", tmp_path / "checks")
    monkeypatch.setattr(sys, "argv", ["x", "--checker", "heuristic", "--classifier", "heuristic"])
    assert evaluate_answer_check.main() == 0
    rep = json.loads((tmp_path / "checks" / "eval_heuristic.json").read_text())
    assert rep["all"]["n"] == 48 and rep["dev"] and rep["test"] and rep["chosen_threshold_on_dev"] is not None


# ---------------------------------------------------------------- repeats
def test_live_benchmark_repeats_use_distinct_run_ids(tmp_path, monkeypatch, toy_registry):
    from jev_router.schemas import VLMResult
    from scripts import run_live_benchmark as rlb

    calls = []

    class FakeExecutor:
        def __init__(self, *a, **k):
            pass

        def run(self, key, question, images, **kw):
            calls.append(key)
            return VLMResult(key=key, model_id=key, backend="openrouter", status="ok", answer="ok", cost_usd=0.0001)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(rlb, "VLMExecutor", FakeExecutor)
    monkeypatch.setattr(rlb, "check_openrouter_key", lambda key: None)
    monkeypatch.setattr(rlb, "LIVE_DIR", tmp_path)
    monkeypatch.setattr("jev_router.classify_cache.CACHE", tmp_path / "cache")
    monkeypatch.setattr(sys, "argv", ["x", "--cases", "GT-E1", "--models", "GEMMA", "--repeats", "2",
                                      "--run-id", "t", "--max-cost", "1"])
    assert rlb.main() == 0
    assert rlb.main() == 0  # resume: nothing new
    rows = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert sorted(r["run_id"] for r in rows) == ["t-r1", "t-r2"] and len(calls) == 2


def test_expired_openrouter_key_stops_before_any_call(tmp_path, monkeypatch, toy_registry):
    from scripts import run_live_benchmark as rlb

    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("no VLM call should be made")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(rlb, "VLMExecutor", Boom)
    monkeypatch.setattr(rlb, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(rlb, "check_openrouter_key", lambda key: "HTTP 401: API key expired.")
    monkeypatch.setattr(sys, "argv", ["x", "--cases", "GT-E1", "--models", "GEMMA", "--run-id", "t"])
    assert rlb.main() == 2 and not (tmp_path / "t.jsonl").exists()
