from __future__ import annotations

import pytest
from langchain_typesafe import ClassifierResponse, NoulAnswer

from jev_router.actions import coordinate_tool, decide_action, haversine_m, parse_coordinates
from jev_router.answer_check import AnswerChecker, heuristic_check


def with_flags(c, **kw):
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_coordinate_tool_distance_and_direction():
    text = "Landmark in Image A: 52.5163 N, 13.3777 E. Landmark in Image B: 52.5186 N, 13.3762 E."
    assert [p[0] for p in parse_coordinates(text)] == ["Image A", "Image B"]
    out = coordinate_tool(text)
    assert "275 m" in out and "Farther north: Image B" in out
    assert coordinate_tool("no coordinates here") is None


def test_haversine_paris_rome():
    assert haversine_m((48.8584, 2.2945), (41.8902, 12.4922)) == pytest.approx(1_109_000, rel=0.01)


def test_use_tools_runs_coordinate_tool(classification_factory, routing_config):
    c = with_flags(classification_factory(task="spatial_reasoning"), needs_external_tools=0.8)
    d = decide_action(c, routing_config, "A: 48.85 N, 2.29 E. B: 41.89 N, 12.49 E.")
    assert d.action == "use_tools" and d.implemented and "Great-circle distance" in d.tool_result


def test_use_tools_without_tool_marks_unimplemented(classification_factory, routing_config):
    c = with_flags(classification_factory(task="factual_retrieval"), needs_external_tools=0.9, answerable_from_inputs=0.1)
    d = decide_action(c, routing_config)
    assert d.action == "use_tools" and not d.implemented and "decline any part" in d.prompt_note


def test_decline_when_unanswerable_and_no_tool(classification_factory, routing_config):
    c = with_flags(classification_factory(), answerable_from_inputs=0.05, needs_external_tools=0.1)
    assert decide_action(c, routing_config).action == "decline"


def test_verify_note_reflects_alignment(classification_factory, routing_config):
    c = with_flags(classification_factory(task="cross_view_matching"), needs_pair_verification=0.9)
    c.input_features = {"n_images": 2, "pair": {"viewpoint_alignment": "unaligned", "keypoints": {"inliers": 3}}}
    d = decide_action(c, routing_config)
    assert d.action == "verify_then_answer" and "could not align" in d.prompt_note


def test_plain_answer(classification_factory, routing_config):
    c = classification_factory()
    c.input_features = {"n_images": 2}
    assert decide_action(c, routing_config).action == "answer"


def test_heuristic_check_orders_risk(classification_factory):
    c = classification_factory(difficulty="hard")
    risky = heuristic_check("The most significant structural change is that a tower was removed and a wing added.",
                            "What changed?", c)
    safe = heuristic_check("No meaningful structural change is visible; only lighting differs.", "What changed?", c)
    assert risky.risk > safe.risk
    assert risky.source == "heuristic_answer_check"


def test_jev_answer_check_sends_images(sample_images, classification_factory):
    calls = []

    class FakeJev:
        def invoke(self, request, config=None):
            calls.append(request)
            return ClassifierResponse(model="jev", answers={
                "unsupported_visual_claim": NoulAnswer(type="noul", noul=0.7),
                "unsupported_fact": NoulAnswer(type="noul", noul=0.1),
                "answers_question": NoulAnswer(type="noul", noul=0.9)})

    res = AnswerChecker(decision_model=FakeJev()).check("A spire was removed.", "What changed?", sample_images,
                                                        classification_factory())
    assert res.source == "jev_answer_check" and res.risk == 0.7
    assert len(calls[0]["state"]["images"]) == 2


def test_answer_check_falls_back_on_error(sample_images, classification_factory):
    class Broken:
        def invoke(self, request, config=None):
            raise TimeoutError("slow")

    res = AnswerChecker(decision_model=Broken()).check("No change.", "What changed?", sample_images,
                                                       classification_factory())
    assert res.source == "heuristic_answer_check" and "slow" in res.error
