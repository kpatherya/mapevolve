from __future__ import annotations

import copy
from typing import Any

import pytest

from jev_router.registry import ModelRegistry, load_routing_config
from jev_router.schemas import LabelDistribution, QueryClassification


def _model(key, price_in, price_out, latency, general, hard, easy, unsupported=0.1, listed="listed", max_images=2, backend="openrouter"):
    def cell(q, n=2, u=unsupported):
        return {"n": n, "mean_score": q * 4, "normalized": q, "correct": 0, "partial": 0, "incorrect": 0, "unsupported_claim_rate": u}

    return {
        "key": key,
        "model_id": f"vendor/{key.lower()}",
        "provider": "vendor",
        "backend": backend,
        "max_output_tokens": None,
        "chat_kwargs": {},
        "availability": {"status": listed},
        "image_input": {"supported": True, "max_images": max_images},
        "pricing": None if price_in is None else {
            "input_usd_per_mtok": price_in, "output_usd_per_mtok": price_out,
            "image_usd_per_image": None, "source": "test",
        },
        "latency": {"mean_seconds": latency, "n": 6},
        "tokens": {"mean_input_tokens_per_query": 2000, "mean_output_tokens_per_query": 200, "images_per_query": 2},
        "observed_cost": {"mean_usd_per_query": 0.01},
        "evaluation": {
            "general": {**cell(general, n=6)},
            "task_scores": {
                "temporal_comparison": {
                    "overall": cell((hard + easy) / 2, n=4),
                    "by_difficulty": {"easy": cell(easy), "hard": cell(hard)},
                }
            },
            "score_sources": ["llm_judge"],
        },
    }


@pytest.fixture
def routing_config() -> dict[str, Any]:
    return copy.deepcopy(load_routing_config())


@pytest.fixture
def toy_registry(routing_config) -> ModelRegistry:
    data = {
        "models": [
            _model("CHEAP", 0.1, 0.3, 5.0, general=0.70, hard=0.30, easy=1.00),
            _model("MID", 0.5, 2.0, 8.0, general=0.75, hard=0.55, easy=0.95),
            _model("PREMIUM", 2.0, 12.0, 20.0, general=0.80, hard=0.80, easy=0.95, unsupported=0.05),
            _model("ONE_IMAGE", 0.05, 0.1, 3.0, general=0.90, hard=0.90, easy=0.90, max_images=1),
            _model("RETIRED", 0.01, 0.01, 1.0, general=0.99, hard=0.99, easy=0.99, listed="not_listed"),
        ]
    }
    return ModelRegistry(data, shrinkage_k=routing_config["registry"]["shrinkage_k"], defaults=routing_config["registry"])


def make_classification(task="temporal_comparison", difficulty="hard", p=1.0, source="jev", confidence=0.9):
    other = {"easy": "hard", "hard": "easy", "moderate": "easy"}[difficulty]
    probs = {difficulty: p} if p >= 1.0 else {difficulty: p, other: 1 - p}
    return QueryClassification(
        query_type=LabelDistribution(task, {task: 1.0}, 0.9),
        difficulty=LabelDistribution(difficulty, probs, confidence),
        visual_expertise=LabelDistribution("architectural_detail", {"architectural_detail": 1.0}, 0.9),
        needs_image_retrieval=0.1,
        needs_pair_verification=0.2,
        needs_external_tools=0.1,
        source=source,
    )


@pytest.fixture
def classification_factory():
    return make_classification


@pytest.fixture
def sample_images():
    from jev_router.benchmark_data import BENCH

    return [BENCH / "data/processed/E1_A.jpg", BENCH / "data/processed/E1_B.jpg"]


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    """Scripts call load_dotenv(), which never overrides a variable that is already set:
    blank keys keep a developer's .env from turning tests into paid API calls."""
    for var in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(var, "")
