from __future__ import annotations

import pytest

from jev_router.policy import RoutingPolicy


def test_shrinkage_toward_general(toy_registry):
    est = toy_registry.score("CHEAP", "temporal_comparison", "hard")
    # (2 * 0.30 + 2 * 0.70) / 4
    assert est.quality == pytest.approx(0.50)
    assert "n=2" in est.source


def test_unknown_task_falls_back_to_general(toy_registry):
    est = toy_registry.score("MID", "spatial_reasoning", "hard")
    assert est.quality == pytest.approx(0.75)
    assert est.source.startswith("general fallback")


def test_filters_images_and_availability(toy_registry, routing_config, classification_factory):
    policy = RoutingPolicy(toy_registry, routing_config)
    decision = policy.route(classification_factory(), n_images=2, mode="accuracy", budget_usd=1.0)
    reasons = {c.key: c.exclusion_reasons for c in decision.candidates}
    assert "cannot take 2 image(s)" in reasons["ONE_IMAGE"]
    assert any("not_listed" in r for r in reasons["RETIRED"])
    assert decision.selected_key not in {"ONE_IMAGE", "RETIRED"}


def test_single_image_query_can_use_single_image_model(toy_registry, routing_config, classification_factory):
    policy = RoutingPolicy(toy_registry, routing_config)
    decision = policy.route(classification_factory(), n_images=1, mode="accuracy", budget_usd=1.0)
    assert decision.selected_key == "ONE_IMAGE"


def test_mode_specific_choices_differ_by_difficulty(toy_registry, routing_config, classification_factory):
    policy = RoutingPolicy(toy_registry, routing_config)
    easy = classification_factory(difficulty="easy")
    hard = classification_factory(difficulty="hard")
    assert policy.route(easy, 2, "economy", 1.0).selected_key == "CHEAP"
    assert policy.route(hard, 2, "accuracy", 1.0).selected_key == "PREMIUM"
    # Same mode, different difficulty: only PREMIUM clears the floor on hard cases.
    assert policy.route(hard, 2, "economy", 1.0).selected_key == "PREMIUM"


def test_budget_excludes_expensive_models(toy_registry, routing_config, classification_factory):
    policy = RoutingPolicy(toy_registry, routing_config)
    premium_cost, _ = toy_registry.estimate_cost("PREMIUM", 2)
    decision = policy.route(classification_factory(), 2, "accuracy", budget_usd=premium_cost / 2)
    assert decision.selected_key != "PREMIUM"
    premium = next(c for c in decision.candidates if c.key == "PREMIUM")
    assert any("budget" in r for r in premium.exclusion_reasons)


def test_economy_relaxes_floor_when_nobody_passes(toy_registry, routing_config, classification_factory):
    routing_config["modes"]["economy"]["min_quality"] = 0.99
    policy = RoutingPolicy(toy_registry, routing_config)
    decision = policy.route(classification_factory(), 2, "economy", 1.0)
    assert any("relaxed" in line for line in decision.rationale)
    assert decision.selected_key == "PREMIUM"


def test_uses_full_difficulty_distribution(toy_registry, routing_config, classification_factory):
    policy = RoutingPolicy(toy_registry, routing_config)
    certain = policy.route(classification_factory(difficulty="hard", p=1.0), 2, "accuracy", 1.0)
    mixed = policy.route(classification_factory(difficulty="hard", p=0.5), 2, "accuracy", 1.0)
    q = lambda d, k: next(c.expected_quality for c in d.candidates if c.key == k)
    assert q(mixed, "CHEAP") > q(certain, "CHEAP")


def test_deterministic(toy_registry, routing_config, classification_factory):
    policy = RoutingPolicy(toy_registry, routing_config)
    runs = {policy.route(classification_factory(), 2, "balanced", 1.0).selected_key for _ in range(5)}
    assert len(runs) == 1


def test_warnings_for_fallback_and_low_confidence(toy_registry, routing_config, classification_factory):
    policy = RoutingPolicy(toy_registry, routing_config)
    decision = policy.route(classification_factory(source="heuristic_fallback", confidence=0.1), 2, "balanced", 1.0)
    text = " ".join(decision.warnings)
    assert "Jev was not used" in text and "Low Jev confidence" in text


def test_no_eligible_model(toy_registry, routing_config, classification_factory):
    policy = RoutingPolicy(toy_registry, routing_config)
    decision = policy.route(classification_factory(), 2, "accuracy", budget_usd=1e-9)
    assert decision.selected_key is None and "no eligible model" in decision.warnings


def test_baselines(toy_registry, routing_config):
    policy = RoutingPolicy(toy_registry, routing_config)
    assert policy.cheapest_eligible(2) == "CHEAP"
    assert policy.strongest(2) == "PREMIUM"
    assert policy.fixed_task("temporal_comparison", 2) == "PREMIUM"


def test_unlisted_model_uses_historical_cost(toy_registry):
    toy_registry.get("MID")["pricing"] = None
    cost, source = toy_registry.estimate_cost("MID", 2)
    assert cost == pytest.approx(0.01) and source.startswith("historical")


def test_invalid_mode(toy_registry, routing_config, classification_factory):
    with pytest.raises(ValueError):
        RoutingPolicy(toy_registry, routing_config).route(classification_factory(), 2, "turbo")
