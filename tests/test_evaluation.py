from __future__ import annotations

import csv
import json
import sys

from jev_router import classifier as classifier_mod
from jev_router.benchmark_data import load_cases, load_responses, model_metrics
from jev_router.dataset import load_all


def test_benchmark_metrics_match_published_summary():
    metrics = model_metrics(load_responses())
    assert len(metrics) == 8
    assert metrics["MISTRAL"]["general"]["mean_score"] == 3.1667
    assert metrics["KIMI"]["general"]["unsupported_claim_rate"] == 0.8333
    assert metrics["GEMMA"]["general"]["partial"] == 2


def test_leave_one_out_drops_the_case():
    records = load_responses()
    full = model_metrics(records)["QWEN"]["task_scores"]["temporal_comparison"]["by_difficulty"]["hard"]
    loo = model_metrics(records, exclude_cases=["H1"])["QWEN"]["task_scores"]["temporal_comparison"]["by_difficulty"]["hard"]
    assert full["n"] == 2 and loo["n"] == 1


def _reference_snippets():
    snippets = [c["ground_truth_raw"][:60] for c in load_cases()]
    for c in load_all():
        snippets.append(c.reference["answer"][:60])
    return snippets


def test_offline_eval_never_passes_ground_truth(tmp_path, monkeypatch):
    seen = []
    real = classifier_mod.JevQueryClassifier.classify

    def spy(self, question, images=(), metadata=None, image_aware=None, evidence=None):
        seen.append(json.dumps({"q": question, "images": [str(i) for i in images], "meta": metadata, "ev": evidence}))
        return real(self, question, images, metadata, image_aware, evidence)

    class NoJev:
        def invoke(self, request, config=None):
            seen.append(json.dumps(request["state"], default=str))
            raise RuntimeError("offline test")

    monkeypatch.setattr(classifier_mod.JevQueryClassifier, "classify", spy)
    monkeypatch.setattr(classifier_mod.JevQueryClassifier, "_get_model", lambda self: NoJev())
    monkeypatch.setattr("jev_router.classify_cache.CACHE", tmp_path / "cache")

    from scripts import evaluate_offline

    monkeypatch.setattr(sys, "argv", ["evaluate_offline", "--classifier", "jev", "--output-dir", str(tmp_path)])
    assert evaluate_offline.main() == 0

    blob = "\n".join(seen)
    assert seen, "classifier was never called"
    for snippet in _reference_snippets():
        assert snippet not in blob
    for case in load_cases():
        assert json.dumps(case["rubric"])[:60] not in blob

    with (tmp_path / "offline_eval_summary.csv").open() as fh:
        strategies = {row["strategy"] for row in csv.DictReader(fh)}
    assert {"always_cheapest", "always_strongest", "fixed_task", "jev_adaptive_economy", "jev_adaptive_balanced",
            "jev_adaptive_accuracy", "jev_accuracy_learned", "jev_cascade_balanced", "oracle"} <= strategies
    # 6 cases x (3 registry modes + 1 learned) traces
    assert (tmp_path / "offline_routing_traces.jsonl").read_text().count("\n") == 24


def test_classifier_harness_runs_on_all_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr("jev_router.classify_cache.CACHE", tmp_path / "cache")
    from scripts import evaluate_classifier

    monkeypatch.setattr(evaluate_classifier, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(sys, "argv", ["evaluate_classifier", "--classifier", "heuristic", "--dataset", "gatech_pilot"])
    assert evaluate_classifier.main() == 0
    report = json.loads((tmp_path / "results" / "classifier_eval_heuristic.json").read_text())
    assert report["overall"]["n"] == 12
    assert 0 <= report["overall"]["action"]["accuracy"] <= 1
