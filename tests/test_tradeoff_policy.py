import json
import tempfile
import unittest
from pathlib import Path

from mapevolve.contracts import QueryInterpretation, RoutingRequest
from mapevolve.pareto_policy import ProfileValidationError
from mapevolve.profile_builder import build_profile, wilson_lower_bound
from mapevolve.tradeoff_policy import (
    NoEligibleModelError,
    TradeoffConfig,
    decide_with_tradeoff_policy,
)

FAMILY = "same_building_verification"


def _payload(extra_models=None, results=None):
    models = {
        "cheap": {"tier": "small", "license": "open", "input_cost_usd_per_1m_tokens": 0.1,
                  "output_cost_usd_per_1m_tokens": 0.4, "tokens_per_image": 500},
        "mid": {"tier": "mid", "license": "proprietary", "input_cost_usd_per_1m_tokens": 1.0,
                "output_cost_usd_per_1m_tokens": 4.0, "tokens_per_image": 800},
        "frontier": {"tier": "frontier", "license": "proprietary", "input_cost_usd_per_1m_tokens": 10.0,
                     "output_cost_usd_per_1m_tokens": 40.0, "tokens_per_image": 1000},
        "dominated": {"tier": "mid", "license": "proprietary", "input_cost_usd_per_1m_tokens": 5.0,
                      "output_cost_usd_per_1m_tokens": 20.0, "tokens_per_image": 1000},
    }
    models.update(extra_models or {})
    accs = {"cheap": (0.80, 0.82), "mid": (0.90, 0.91), "frontier": (0.93, 0.94), "dominated": (0.85, 0.85)}
    rows = results
    if rows is None:
        rows = [
            {"family": FAMILY, "benchmark": b, "model_id": m, "score": accs[m][i], "n": 500}
            for m in accs
            for i, b in enumerate(("bench_a", "bench_b"))
        ]
    bench = {b: {"task": "matching", "metric": "accuracy", "split": "test", "source": "x", "date": "2026-01-01"}
             for b in ("bench_a", "bench_b")}
    return {"schema_version": 2, "profile_version": "t1", "benchmarks": bench, "models": models, "results": rows}


class TradeoffPolicyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        a, b = Path(self.dir.name, "a.jpg"), Path(self.dir.name, "b.jpg")
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        self.request = RoutingRequest(request_id="r1", query="Are these the same building?", images=[str(a), str(b)])
        self.interp = QueryInterpretation(request_id="r1", family=FAMILY, rule_id="t", confidence=1.0)
        self.profile = build_profile(_payload())

    def _pick(self, **kw):
        sel = decide_with_tradeoff_policy(self.request, self.interp, self.profile, TradeoffConfig(**kw))
        return sel.decision.backend_id, sel

    def test_tradeoff_parameter_changes_selected_model(self):
        self.assertEqual(self._pick(mode="utility", lam=0.0)[0], "frontier")
        self.assertEqual(self._pick(mode="utility", lam=5.0)[0], "cheap")

    def test_dominated_model_not_on_frontier(self):
        _, sel = self._pick(mode="utility", lam=0.0)
        self.assertNotIn("dominated", [m.model_id for m in sel.pareto_frontier])

    def test_target_picks_cheapest_meeting_accuracy(self):
        self.assertEqual(self._pick(mode="target", min_accuracy=0.86, use_lower_bound=False)[0], "mid")

    def test_target_unreachable_raises(self):
        with self.assertRaises(NoEligibleModelError):
            self._pick(mode="target", min_accuracy=0.99, use_lower_bound=False)

    def test_budget_picks_best_affordable_and_raises_when_none(self):
        _, sel = self._pick(mode="utility", lam=0.0)
        costs = {f["model_id"]: f["cost_usd"] for f in sel.policy_details["frontier"]}
        picked, _ = self._pick(mode="budget", max_cost_usd=costs["mid"] * 1.01)
        self.assertEqual(picked, "mid")
        with self.assertRaises(NoEligibleModelError):
            self._pick(mode="budget", max_cost_usd=0.0)

    def test_details_record_reproducibility_fields(self):
        _, sel = self._pick(mode="utility", lam=0.1)
        d = sel.policy_details
        self.assertEqual(sel.profile_version, "t1")
        self.assertEqual(d["tradeoff"]["mode"], "utility")
        self.assertIn("frontier", d)
        self.assertEqual(d["benchmarks"], ["bench_a", "bench_b"])
        json.dumps(d)

    def test_incomplete_benchmark_coverage_excludes_model(self):
        rows = _payload()["results"]
        rows = [r for r in rows if not (r["model_id"] == "frontier" and r["benchmark"] == "bench_b")]
        profile = build_profile(_payload(results=rows))
        sel = decide_with_tradeoff_policy(self.request, self.interp, profile, TradeoffConfig(mode="utility", lam=0.0))
        self.assertEqual(sel.policy_details["excluded_incomplete_coverage"], ["frontier"])
        self.assertEqual(sel.decision.backend_id, "mid")

    def test_non_executable_requests_skip_policy(self):
        interp = QueryInterpretation(request_id="r1", family="change_captioning", rule_id="t", confidence=1.0)
        sel = decide_with_tradeoff_policy(self.request, interp, self.profile, TradeoffConfig())
        self.assertEqual(sel.decision.action, "abstain")
        self.assertIsNone(sel.selected_model)

    def test_lower_bound_penalises_small_samples(self):
        self.assertLess(wilson_lower_bound(0.9, 20), wilson_lower_bound(0.9, 2000))

    def test_invalid_configs_rejected(self):
        for kw in ({"mode": "budget"}, {"mode": "target"}, {"mode": "x"}, {"mode": "utility", "lam": -1}):
            with self.assertRaises(ProfileValidationError):
                TradeoffConfig(**kw)

    def test_non_accuracy_metric_rejected(self):
        p = _payload()
        p["benchmarks"]["bench_a"]["metric"] = "f1"
        with self.assertRaises(ProfileValidationError):
            build_profile(p)

    def test_coverage_gate_fails_on_thin_profile(self):
        with self.assertRaises(ProfileValidationError):
            self.profile.require_coverage(FAMILY, min_models=10, min_benchmarks=3)
        self.profile.require_coverage(FAMILY, min_models=4, min_benchmarks=2)

    def test_unknown_result_reference_rejected(self):
        rows = [{"family": FAMILY, "benchmark": "nope", "model_id": "cheap", "score": 0.5, "n": 5}]
        with self.assertRaises(ProfileValidationError):
            build_profile(_payload(results=rows))


if __name__ == "__main__":
    unittest.main()
