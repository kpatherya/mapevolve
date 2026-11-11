from __future__ import annotations

import copy

from langchain_core.messages import AIMessage

from jev_router.answer_check import CheckResult
from jev_router.classifier import JevQueryClassifier
from jev_router.pipeline import AdaptiveRouter
from jev_router.policy import RoutingPolicy
from jev_router.quality_model import QualityModel, TrainingRow, model_attributes
from jev_router.vlm import VLMExecutor, parse_structured

from tests.conftest import make_classification


def synthetic_rows():
    """EASYBOT wins easy cases, HARDBOT wins hard ones."""
    rows = []
    for i in range(12):
        hard = i % 2 == 0
        c = make_classification(difficulty="hard" if hard else "easy")
        rows.append(TrainingRow(f"c{i}", "EASYBOT", 1.0 if hard else 4.0, hard, c))
        rows.append(TrainingRow(f"c{i}", "HARDBOT", 3.5 if hard else 3.0, False, c))
    return rows


ATTRS = {"EASYBOT": {"log_price_in": -1.0, "log_price_out": -0.5}, "HARDBOT": {"log_price_in": 0.3, "log_price_out": 1.0}}


def test_predictor_learns_difficulty_specific_winner():
    qm = QualityModel.fit(synthetic_rows(), ATTRS, lam=0.3, n_boot=10)
    hard, easy = make_classification(difficulty="hard"), make_classification(difficulty="easy")
    assert qm.predict(hard, "HARDBOT").quality > qm.predict(hard, "EASYBOT").quality
    assert qm.predict(easy, "EASYBOT").quality > qm.predict(easy, "HARDBOT").quality
    assert qm.predict(hard, "EASYBOT").unsupported > qm.predict(hard, "HARDBOT").unsupported
    assert qm.predict(hard, "HARDBOT").quality_std >= 0


def test_predictor_handles_unseen_model_and_round_trips(tmp_path):
    qm = QualityModel.fit(synthetic_rows(), ATTRS, lam=1.0, n_boot=5)
    pred = qm.predict(make_classification(), "NEWBOT", {"log_price_in": 0.0, "log_price_out": 0.0})
    assert 0 <= pred.quality <= 1 and "unseen model" in pred.source
    qm.save(tmp_path / "qm.json")
    again = QualityModel.load(tmp_path / "qm.json")
    c = make_classification()
    assert abs(again.predict(c, "HARDBOT").quality - qm.predict(c, "HARDBOT").quality) < 1e-5


def test_policy_gates_predictor_on_case_count(toy_registry, routing_config):
    attrs = model_attributes(toy_registry.data["models"])
    rows = [TrainingRow(f"c{i}", k, 3.0, False, make_classification()) for i in range(3) for k in ("CHEAP", "PREMIUM")]
    qm = QualityModel.fit(rows, attrs, n_boot=3)
    qm.meta["query_types"] = ["temporal_comparison"]
    assert RoutingPolicy(toy_registry, routing_config, quality_model=qm).quality_model is None  # 3 < 30
    forced = RoutingPolicy(toy_registry, routing_config, quality_model=qm, use_quality_model=True)
    d = forced.route(make_classification(), 2, "accuracy", 1.0)
    chosen = next(c for c in d.candidates if c.key == d.selected_key)
    assert any("learned predictor" in s for s in chosen.score_sources)


def test_parse_structured():
    assert parse_structured('```json\n{"answer": "Yes.", "evidence": ["x"]}\n```')["answer"] == "Yes."
    assert parse_structured("plain text answer") is None


class FakeChat:
    def __init__(self, key, log, text):
        self.key, self.log, self.text = key, log, text

    def invoke(self, messages):
        self.log.append((self.key, messages))
        return AIMessage(content=self.text.get(self.key, '{"answer": "ok", "evidence": [], "cannot_determine": [], "confidence": 0.8}'),
                         usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                         response_metadata={"finish_reason": "stop"})


def make_router(toy_registry, routing_config, tmp_path, checker=None, decision_model=None, texts=None):
    cfg = copy.deepcopy(routing_config)
    cfg["quality_model"]["path"] = None
    log = []
    executor = VLMExecutor(toy_registry, cfg["generation"], lambda m, g: FakeChat(m["key"], log, texts or {}), "SYS")

    class NoJev:
        def invoke(self, request, config=None):
            raise ConnectionError("offline")

    router = AdaptiveRouter(classifier=JevQueryClassifier(decision_model=decision_model or NoJev()),
                            registry=toy_registry, executor=executor, config=cfg, trace_dir=tmp_path,
                            checker=checker)
    return router, log


def test_evidence_and_action_note_reach_the_vlm(toy_registry, routing_config, sample_images, tmp_path):
    router, log = make_router(toy_registry, routing_config, tmp_path)
    ev = "Landmark in Image A: 48.8584 N, 2.2945 E. Landmark in Image B: 41.8902 N, 12.4922 E."
    trace = router.answer("Which landmark is farther north?", sample_images, "economy", 1.0, evidence=ev)
    assert trace["action"]["action"] == "use_tools"
    texts = [b["text"] for b in log[0][1][1].content if b["type"] == "text"]
    assert any(t.startswith("Supplied evidence:") for t in texts)
    assert any("Coordinate tool" in t for t in texts)
    assert "Task: answer the spatial question" in log[0][1][0].content
    assert trace["result"]["structured"]["answer"] == "ok"


def test_decline_skips_vlm(toy_registry, routing_config, sample_images, tmp_path, monkeypatch):
    router, log = make_router(toy_registry, routing_config, tmp_path)
    real = router.classifier.classify

    def low_answerable(*a, **k):
        c = real(*a, **k)
        c.answerable_from_inputs, c.needs_external_tools = 0.05, 0.1
        return c

    monkeypatch.setattr(router.classifier, "classify", low_answerable)
    trace = router.answer("What is this?", sample_images[:1], "balanced", 1.0)
    assert trace["action"]["action"] == "decline"
    assert trace["result"]["status"] == "declined" and log == []


def test_cascade_escalates_and_keeps_lower_risk(toy_registry, routing_config, sample_images, tmp_path):
    class Checker:
        def check(self, answer, question, images=(), classification=None, evidence=None):
            return CheckResult(0.9 if "risky" in answer else 0.1, "test")

    texts = {"CHEAP": '{"answer": "risky claim"}'}
    router, log = make_router(toy_registry, routing_config, tmp_path, checker=Checker(), texts=texts)
    trace = router.answer("What changed between these two images?", sample_images, "economy", 1.0, cascade=True)
    cz = trace["cascade"]
    assert cz["triggered"] and cz["first_key"] == "CHEAP" and cz["escalated_key"] != "CHEAP"
    assert trace["result"]["key"] == cz["escalated_key"] and len(log) == 2


def test_cascade_off_means_single_call(toy_registry, routing_config, sample_images, tmp_path):
    class Checker:
        def check(self, *a, **k):
            return CheckResult(0.99, "test")

    router, log = make_router(toy_registry, routing_config, tmp_path, checker=Checker())
    trace = router.answer("What changed between these two images?", sample_images, "economy", 1.0, cascade=False)
    assert trace["cascade"] is None and len(log) == 1
