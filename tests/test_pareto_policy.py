from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mapevolve.contracts import RoutingRequest
from mapevolve.pareto_policy import (
    DEFAULT_PROFILES_PATH,
    ProfileValidationError,
    choose_model,
    decide_with_pareto_policy,
    load_profiles,
    pareto_frontier,
)
from mapevolve.query_classifier import interpret_request


class ParetoPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.image_a = Path(self.tmpdir.name) / "a.png"
        self.image_b = Path(self.tmpdir.name) / "b.png"
        png_bytes = b"\x89PNG\r\n\x1a\n"
        self.image_a.write_bytes(png_bytes)
        self.image_b.write_bytes(png_bytes)

    def _request(self, query: str, image_count: int) -> RoutingRequest:
        images = []
        if image_count >= 1:
            images.append(str(self.image_a))
        if image_count >= 2:
            images.append(str(self.image_b))
        return RoutingRequest(
            request_id="req-1",
            query=query,
            images=images,
        )

    def test_load_profiles_default_file(self) -> None:
        catalog = load_profiles(DEFAULT_PROFILES_PATH)
        same_building = catalog.for_family("same_building_verification")
        self.assertGreaterEqual(len(same_building.models), 2)
        self.assertEqual(catalog.profile_version, "mvp_v1")

    def test_pareto_frontier_excludes_dominated_models(self) -> None:
        catalog = load_profiles(DEFAULT_PROFILES_PATH)
        same_building = catalog.for_family("same_building_verification")
        model_ids = [model.model_id for model in same_building.models]
        self.assertIn("legacy/vision-pro", model_ids)

        frontier = pareto_frontier(same_building.models)
        frontier_ids = [model.model_id for model in frontier]
        self.assertNotIn("legacy/vision-pro", frontier_ids)
        self.assertEqual(
            frontier_ids,
            [
                "openai/gpt-4.1-mini",
                "anthropic/claude-3.5-sonnet",
            ],
        )

    def test_choose_model_uses_accuracy_margin_numerically(self) -> None:
        catalog = load_profiles(DEFAULT_PROFILES_PATH)
        frontier = pareto_frontier(catalog.for_family("same_building_verification").models)

        strict_choice = choose_model(frontier, accuracy_margin=0.0)
        self.assertEqual(strict_choice.model_id, "anthropic/claude-3.5-sonnet")

        relaxed_choice = choose_model(frontier, accuracy_margin=0.02)
        self.assertEqual(relaxed_choice.model_id, "openai/gpt-4.1-mini")

    def test_decide_execute_for_same_building_with_two_images(self) -> None:
        request = self._request("Are these the same building?", image_count=2)
        interpretation = interpret_request(request)
        selection = decide_with_pareto_policy(request, interpretation)
        self.assertEqual(selection.decision.action, "execute")
        self.assertEqual(selection.decision.reason_code, "ready")
        self.assertEqual(selection.decision.backend_id, "openai/gpt-4.1-mini")
        self.assertEqual(selection.profile_version, "mvp_v1")
        self.assertTrue(selection.pareto_frontier)

    def test_preserves_request_evidence_without_profile_lookup(self) -> None:
        request = self._request("Are these the same building?", image_count=1)
        interpretation = interpret_request(request)
        selection = decide_with_pareto_policy(request, interpretation)
        self.assertEqual(selection.decision.action, "request_evidence")
        self.assertIsNone(selection.selected_model)
        self.assertEqual(selection.pareto_frontier, ())
        self.assertIsNone(selection.profile_version)

    def test_preserves_unsupported_family_abstention_without_profile_lookup(self) -> None:
        request = self._request("What changed between these photos?", image_count=2)
        interpretation = interpret_request(request)
        selection = decide_with_pareto_policy(request, interpretation)
        self.assertEqual(selection.decision.action, "abstain")
        self.assertEqual(selection.decision.reason_code, "capability_not_yet_supported")
        self.assertIsNone(selection.selected_model)
        self.assertEqual(selection.pareto_frontier, ())

    def test_preserves_unknown_family_abstention_without_profile_lookup(self) -> None:
        request = self._request("Tell me anything useful.", image_count=0)
        interpretation = interpret_request(request)
        selection = decide_with_pareto_policy(request, interpretation)
        self.assertEqual(selection.decision.action, "abstain")
        self.assertEqual(selection.decision.reason_code, "unknown_query_family")
        self.assertIsNone(selection.selected_model)
        self.assertEqual(selection.pareto_frontier, ())

    def test_missing_required_profile_fields_fail_explicitly(self) -> None:
        bad_payload = {
            "schema_version": 1,
            "families": {},
        }
        path = Path(self.tmpdir.name) / "bad_profiles.json"
        path.write_text(json.dumps(bad_payload), encoding="utf-8")
        with self.assertRaises(ProfileValidationError):
            load_profiles(path)

    def test_unsupported_profile_family_fails_explicitly(self) -> None:
        payload = json.loads(DEFAULT_PROFILES_PATH.read_text(encoding="utf-8"))
        payload["families"]["made_up_family"] = {"recommended_model": None, "models": []}
        path = Path(self.tmpdir.name) / "bad_family_profiles.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ProfileValidationError):
            load_profiles(path)


if __name__ == "__main__":
    unittest.main()
