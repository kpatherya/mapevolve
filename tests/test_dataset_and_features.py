from __future__ import annotations

import copy

import pytest

from jev_router.dataset import Case, coverage, load_all, routing_inputs, validate
from jev_router.pair_features import alignment_label, pair_features


@pytest.fixture(scope="module")
def cases():
    return load_all()


def test_committed_manifests_are_valid(cases):
    assert validate(cases) == []
    assert {c.dataset for c in cases} == {"gatech_pilot", "demo_world"}


def test_split_leakage_is_detected(cases):
    a = next(c for c in cases if c.dataset == "demo_world")
    b = copy.deepcopy(a)
    b.case_id, b.split = a.case_id + "-copy", ("test" if a.split == "dev" else "dev")
    errors = validate([a, b], check_files=False)
    assert any("appears in splits" in e for e in errors)


def test_unanswerable_cannot_expect_answer(cases):
    c = copy.deepcopy(next(c for c in cases if not c.answerable))
    c.expected_action = "answer"
    assert any("unanswerable" in e for e in validate([c], check_files=False))


def test_routing_inputs_exclude_labels(cases):
    for c in cases:
        inp = routing_inputs(c)
        assert set(inp) == {"question", "images", "supplied_evidence"}


def test_coverage_counts_every_case(cases):
    report = coverage(cases, min_per_cell=1)
    total = sum(n for row in report["grid"].values() for d in row.values() for n in d.values())
    assert total == len(cases)


def test_pair_features_identical_image_is_aligned(sample_images):
    f = pair_features([sample_images[0], sample_images[0]])
    assert f["viewpoint_alignment"] == "aligned"
    assert f["histogram_similarity"] > 0.99 and f["dhash_similarity"] == 1.0


def test_pair_features_none_for_single_image(sample_images):
    assert pair_features(sample_images[:1]) is None


def test_alignment_thresholds():
    assert alignment_label(100, 0.8) == "aligned"
    assert alignment_label(12, 0.2) == "partially_aligned"
    assert alignment_label(3, 0.9) == "unaligned"


def test_case_round_trip(cases):
    c = cases[0]
    assert Case.from_dict(c.__dict__) == c
