from __future__ import annotations

import json
import sys

import pytest
import yaml

from jev_router.offline import deep_merge, fixed_choice, load_offline, route_all, score_choices
from jev_router.pareto import (Obs, analyze_cell, cell_lookup, complete_cases, dominates, frontier,
                               quality_cost_front, summarize)
from jev_router.policy import RoutingPolicy
from jev_router.registry import load_routing_config

from tests.conftest import make_classification

OBJ = [("quality", "max"), ("cost_usd", "min")]


def obs(case, model, score, cost, latency=1.0, unsupported=False, answerable=True, run="r1"):
    return Obs(case, model, score, unsupported, cost, latency, answerable, run)


def test_dominance_rules():
    a = {"quality": 0.9, "cost_usd": 1.0}
    assert dominates(a, {"quality": 0.8, "cost_usd": 1.0}, OBJ)
    assert not dominates(a, {"quality": 0.95, "cost_usd": 2.0}, OBJ)   # trade-off
    assert not dominates(a, dict(a), OBJ)                             # equal is not domination


def test_frontier_and_dominated_by():
    rows = [obs("c1", "CHEAP", 3, 0.001), obs("c1", "PRICEY", 4, 0.01), obs("c1", "BAD", 2, 0.02)]
    points = summarize(rows)
    on = frontier(points, OBJ)
    assert on == {"CHEAP", "PRICEY"}
    assert set(points["BAD"].dominated_by) == {"CHEAP", "PRICEY"}


def test_bootstrap_membership_and_optional_axes():
    rows = []
    for i in range(8):
        rows += [obs(f"c{i}", "A", 4 if i % 2 else 2, 0.001), obs(f"c{i}", "B", 3, 0.0005),
                 obs(f"c{i}", "C", 1, 0.01, unsupported=True)]
    # repeated run for A on one case -> inconsistency becomes available only if every model has repeats
    cell = analyze_cell(rows, n_boot=100)
    pts = cell["points"]
    assert pts["C"]["membership"] == 0.0 and not pts["C"]["on_frontier"]
    assert 0 < pts["A"]["membership"] <= 1
    assert ("inconsistency", "min") not in cell["objectives"]


def test_unanswerable_and_repeat_metrics():
    rows = [obs("u1", "A", 4, 0.001, answerable=False), obs("u1", "A", 2, 0.001, answerable=False, run="r2"),
            obs("a1", "A", 4, 0.001)]
    m = summarize(rows)["A"].metrics
    assert m["unanswerable_quality"] == pytest.approx(0.75)
    assert m["inconsistency"] == pytest.approx(0.25)


def test_complete_cases_keeps_only_fully_graded():
    rows = [obs("c1", "A", 4, 1), obs("c1", "B", 3, 1), obs("c2", "A", 4, 1)]
    assert {o.case_id for o in complete_cases(rows)} == {"c1"}


def test_quality_cost_front():
    front = quality_cost_front([("a", 0.7, 0.001), ("b", 0.8, 0.002), ("c", 0.75, 0.003), ("d", 0.9, 0.01)])
    assert [n for n, _, _ in front] == ["a", "b", "d"]


def test_cell_lookup_falls_back_to_broader_cells():
    table = {"cells": {"type=x|difficulty=hard": {"n_cases": 2}, "type=x": {"n_cases": 6}, "all": {"n_cases": 9}}}
    assert cell_lookup(table, "x", "hard", 5)[0] == "type=x"
    assert cell_lookup(table, "y", "easy", 5)[0] == "all"
    assert cell_lookup(table, "y", "easy", 50) is None


def _table(points):
    return {"cells": {"all": {"n_cases": 10, "points": {k: {"membership": v} for k, v in points.items()}}}}


def test_policy_prunes_dominated_but_never_everything(toy_registry, routing_config):
    cfg = deep_merge(routing_config, {"pareto": {"prune_dominated": True, "table": None, "min_cases": 5,
                                                 "min_membership": 0.2}})
    pol = RoutingPolicy(toy_registry, cfg)
    pol.pareto_table = _table({"CHEAP": 0.9, "MID": 0.05, "PREMIUM": 0.6})
    d = pol.route(make_classification(), 2, "accuracy", 1.0)
    mid = next(c for c in d.candidates if c.key == "MID")
    assert not mid.eligible and any("Pareto-dominated" in r for r in mid.exclusion_reasons)
    assert any("Pruned dominated" in r for r in d.rationale)

    pol.pareto_table = _table({"CHEAP": 0.0, "MID": 0.0, "PREMIUM": 0.0})
    d = pol.route(make_classification(), 2, "accuracy", 1.0)
    assert d.selected_key is not None
    assert any("would remove every eligible model" in w for w in d.warnings)


def test_pruning_off_by_default(toy_registry, routing_config):
    assert RoutingPolicy(toy_registry, routing_config).pareto_table is None


def test_deep_merge_and_overlay(tmp_path):
    base = {"modes": {"economy": {"min_quality": 0.7, "x": 1}}, "a": 1}
    out = deep_merge(base, {"modes": {"economy": {"min_quality": 0.6}}})
    assert out["modes"]["economy"] == {"min_quality": 0.6, "x": 1} and base["modes"]["economy"]["min_quality"] == 0.7
    ov = tmp_path / "ov.yaml"
    ov.write_text(yaml.safe_dump({"registry": {"shrinkage_k": 4}}))
    cfg = load_routing_config(overlay=ov)
    assert cfg["registry"]["shrinkage_k"] == 4 and cfg["_overlay"] == str(ov)


def test_registry_split_excludes_other_split_grades(tmp_path, monkeypatch):
    monkeypatch.setattr("jev_router.classify_cache.CACHE", tmp_path / "cache")
    monkeypatch.setattr("jev_router.graded.GRADED_DIR", tmp_path / "graded")  # pilot only
    dev = load_offline("heuristic", split="dev", registry_split="dev")
    assert dev.case_ids and all(dev.manifest[c].split == "dev" for c in dev.case_ids)
    assert {r.case_id for r in dev.records} <= set(dev.case_ids)
    # LOO: the routed case never informs its own registry
    reg = dev.registry_for(dev.case_ids[0])
    assert reg.models  # built
    routes = route_all(dev, dev.cfg, "balanced")
    assert set(routes) == set(dev.case_ids)
    s = score_choices(dev, fixed_choice(dev, dev.pool[0]))
    assert s["n_cases"] == len(dev.case_ids)


def test_tuner_keeps_overlay_out_of_config_with_few_cases(tmp_path, monkeypatch):
    from scripts import tune_router

    monkeypatch.setattr("jev_router.classify_cache.CACHE", tmp_path / "cache")
    monkeypatch.setattr(tune_router, "OUT", tmp_path / "tuning")
    monkeypatch.setattr(tune_router, "TUNED", tmp_path / "routing.tuned.yaml")
    monkeypatch.setattr(tune_router, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["tune_router", "--classifier", "heuristic"])
    assert tune_router.main() == 0
    assert not (tmp_path / "routing.tuned.yaml").exists()
    assert (tmp_path / "tuning" / "candidate_overlay.yaml").exists()
    summary = json.loads((tmp_path / "tuning" / "summary.json").read_text())
    assert all(m.startswith("GT-") for m in summary["dev_cases"])
    # ties prefer current settings: accuracy should keep the default penalty and k
    base = load_routing_config()
    acc = summary["modes"]["accuracy"]["best"]
    assert acc["modes.accuracy.unsupported_penalty"] == base["modes"]["accuracy"]["unsupported_penalty"]
    assert acc["modes.accuracy.shrinkage_k"] == base["registry"]["shrinkage_k"]  # ties keep the default k


def test_pareto_report_runs(tmp_path, monkeypatch):
    from scripts import pareto_report

    monkeypatch.setattr("jev_router.classify_cache.CACHE", tmp_path / "cache")
    monkeypatch.setattr(pareto_report, "OUT", tmp_path / "pareto")
    monkeypatch.setattr(pareto_report, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["pareto_report", "--classifier", "heuristic", "--n-boot", "20"])
    assert pareto_report.main() == 0
    table = json.loads((tmp_path / "pareto" / "frontier.json").read_text())
    assert "all" in table["cells"] and table["cells"]["all"]["frontier"]
    html = (tmp_path / "pareto" / "index.html").read_text()
    assert "<svg" in html and "router setting" in html


def test_svg_embeds_theme_colors():
    from scripts.pareto_report import svg_chart

    svg = svg_chart("t", [("A", 0.8, 0.001, True), ("B", 0.6, 0.01, False)], [], theme="dark")
    assert "<style>" in svg and "#3987e5" in svg and "router setting" not in svg
    assert svg_chart("t", [], []) == ""


def test_dominance_skips_missing_metrics():
    from jev_router.pareto import dominates

    objs = [("quality", "max"), ("unanswerable_quality", "max")]
    assert dominates({"quality": 0.9, "unanswerable_quality": None}, {"quality": 0.5, "unanswerable_quality": None}, objs)


def test_well_covered_models_ignores_sparse_models():
    from jev_router.pareto import well_covered_models

    pairs = [(f"c{i}", m) for i in range(10) for m in ("A", "B")] + [("c0", "PILOT_ONLY")]
    assert well_covered_models(pairs) == ["A", "B"]
