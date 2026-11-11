from __future__ import annotations

import pytest
from langchain_typesafe import ChoiceAnswer, ClassifierResponse, NoulAnswer, ScoreAnswer

from jev_router.classifier import JevQueryClassifier, build_state, heuristic_classify


def fake_response() -> ClassifierResponse:
    return ClassifierResponse(
        model="jev-test",
        request_id="req-123",
        usage={"input_tokens": 120, "output_tokens": 6},
        answers={
            "query_type": ChoiceAnswer(
                type="choice", choice="temporal_comparison",
                probabilities={"temporal_comparison": 0.8, "cross_view_matching": 0.15, "landmark_identification": 0.05},
                confidence=0.7,
            ),
            "difficulty": ScoreAnswer(
                type="score", score=1.6, legend={0: "easy", 1: "moderate", 2: "hard"},
                probabilities={0: 0.1, 1: 0.2, 2: 0.7}, confidence=0.5,
            ),
            "visual_expertise": ChoiceAnswer(
                type="choice", choice="historical_photo_interpretation",
                probabilities={"historical_photo_interpretation": 0.9, "architectural_detail": 0.1},
                confidence=0.8,
            ),
            "needs_image_retrieval": NoulAnswer(type="noul", noul=0.1),
            "needs_pair_verification": NoulAnswer(type="noul", noul=0.75),
            "needs_external_tools": NoulAnswer(type="noul", noul=0.2),
            "answerable_from_inputs": NoulAnswer(type="noul", noul=0.9),
        },
    )


class FakeJev:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.response, self.error = response, error

    def invoke(self, request, config=None):
        self.calls.append(request)
        if self.error:
            raise self.error
        return self.response


def test_single_invocation_with_all_typed_questions(sample_images):
    jev = FakeJev(fake_response())
    result = JevQueryClassifier(decision_model=jev).classify("What changed?", sample_images)
    assert len(jev.calls) == 1
    questions = jev.calls[0]["questions"]
    assert set(questions) == {
        "query_type", "difficulty", "visual_expertise",
        "needs_image_retrieval", "needs_pair_verification", "needs_external_tools",
        "answerable_from_inputs",
    }
    assert {type(q).__name__ for q in questions.values()} == {"Choice", "Score", "Noul"}
    assert result.source == "jev"
    assert result.query_type.label == "temporal_comparison"
    assert result.difficulty.label == "hard"
    assert result.difficulty.probabilities == {"easy": 0.1, "moderate": 0.2, "hard": 0.7}
    assert result.difficulty.confidence == 0.5
    assert result.needs_pair_verification == 0.75
    assert result.flags()["needs_pair_verification"] is True
    assert result.request_id == "req-123"
    assert result.answerable_from_inputs == 0.9
    assert result.raw["answers"]["query_type"]["probabilities"]["temporal_comparison"] == 0.8


def test_state_is_text_and_metadata_only_by_default(sample_images):
    jev = FakeJev(fake_response())
    JevQueryClassifier(decision_model=jev).classify("What changed?", sample_images)
    state = jev.calls[0]["state"]
    assert set(state) == {"question", "num_images", "image_metadata", "pair_features"}
    assert state["num_images"] == 2
    assert state["pair_features"]["viewpoint_alignment"] in {"aligned", "partially_aligned", "unaligned"}


def test_supplied_evidence_reaches_jev(sample_images):
    jev = FakeJev(fake_response())
    JevQueryClassifier(decision_model=jev).classify("Which is farther north?", sample_images, evidence="A: 48.8 N, 2.3 E")
    assert jev.calls[0]["state"]["supplied_evidence"] == "A: 48.8 N, 2.3 E"


def test_image_aware_mode_attaches_jpeg_blocks(sample_images):
    jev = FakeJev(fake_response())
    out = JevQueryClassifier(decision_model=jev).classify("What changed?", sample_images, image_aware=True)
    blocks = jev.calls[0]["state"]["images"]
    assert len(blocks) == 2 and all(b["mime_type"] == "image/jpeg" for b in blocks)
    assert out.image_aware is True


def test_fallback_on_jev_error(sample_images):
    jev = FakeJev(error=TimeoutError("jev timed out"))
    result = JevQueryClassifier(decision_model=jev).classify("What changed between these two images?", sample_images)
    assert result.source == "heuristic_fallback"
    assert "jev timed out" in result.error
    assert result.query_type.label == "temporal_comparison"


def test_answer_bearing_metadata_is_rejected():
    with pytest.raises(ValueError, match="answer-bearing"):
        build_state("What changed?", [], metadata={"ground_truth_raw": "No change."})


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Who funded the creation of this building?", "factual_retrieval"),
        ("Which of these two buildings is closer to Klaus?", "spatial_reasoning"),
        ("Do these two photos show the same building?", "cross_view_matching"),
        ("Compare these two images of the same building. What changed?", "temporal_comparison"),
        ("What building is this?", "landmark_identification"),
        ("What does the inscription above the entrance say?", "factual_retrieval"),
        ("Is the clock tower at the left end or the right end?", "spatial_reasoning"),
    ],
)
def test_heuristic_query_types(question, expected):
    assert heuristic_classify(question, image_info=[{}, {}]).query_type.label == expected


def test_heuristic_marks_monochrome_pairs_hard():
    info = [{"likely_monochrome": True}, {"likely_monochrome": False}]
    assert heuristic_classify("What changed?", image_info=info).difficulty.label == "hard"


def test_heuristic_uses_alignment_and_evidence():
    aligned = {"viewpoint_alignment": "aligned"}
    assert heuristic_classify("What changed?", image_info=[{}, {}], pair=aligned).difficulty.label == "easy"
    q = "Which landmark is farther north?"
    with_coords = heuristic_classify(q, image_info=[{}, {}], pair={}, evidence="A: 48.85 N, 2.29 E. B: 41.89 N, 12.49 E.")
    without = heuristic_classify(q, image_info=[{}, {}], pair={})
    assert with_coords.answerable_from_inputs > 0.5 > without.answerable_from_inputs
