from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mapevolve.contracts import RoutingRequest
from mapevolve.query_classifier import classify_and_decide, interpret_request, non_execution_decision


class QueryClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.image_a = Path(self.tmpdir.name) / "a.png"
        self.image_b = Path(self.tmpdir.name) / "b.png"
        self.image_c = Path(self.tmpdir.name) / "c.png"
        png_bytes = b"\x89PNG\r\n\x1a\n"
        self.image_a.write_bytes(png_bytes)
        self.image_b.write_bytes(png_bytes)
        self.image_c.write_bytes(png_bytes)

    def _request(self, query: str, images: list[str]) -> RoutingRequest:
        return RoutingRequest(
            request_id="req-1",
            query=query,
            images=images,
        )

    def test_classifies_same_building_verification(self) -> None:
        interpretation = interpret_request(
            self._request("Are these the same building?", [str(self.image_a), str(self.image_b)])
        )
        self.assertEqual(interpretation.family, "same_building_verification")
        self.assertEqual(interpretation.rule_id, "rule.same_building.keyword_v1")

    def test_classifies_change_captioning(self) -> None:
        interpretation = interpret_request(
            self._request(
                "Identify the similarities and differences between these photos.", [str(self.image_a)]
            )
        )
        self.assertEqual(interpretation.family, "change_captioning")

    def test_classifies_evidence_backed_investigation(self) -> None:
        interpretation = interpret_request(
            self._request("Who founded this building and when was it built?", [])
        )
        self.assertEqual(interpretation.family, "evidence_backed_investigation")

    def test_classifies_proximity_computation(self) -> None:
        interpretation = interpret_request(
            self._request("How close is this building to city hall?", [str(self.image_a)])
        )
        self.assertEqual(interpretation.family, "proximity_computation")

    def test_ambiguous_query_maps_to_unknown(self) -> None:
        interpretation = interpret_request(self._request("What can you tell me about this?", []))
        self.assertEqual(interpretation.family, "unknown")
        self.assertEqual(interpretation.rule_id, "rule.unknown.fallback_v1")

    def test_multi_match_query_uses_first_rule_precedence(self) -> None:
        interpretation = interpret_request(
            self._request(
                "Are these the same building? Also list similarities and differences.",
                [str(self.image_a), str(self.image_b)],
            )
        )
        self.assertEqual(interpretation.family, "same_building_verification")
        self.assertEqual(interpretation.rule_id, "rule.same_building.keyword_v1")

    def test_classification_is_independent_of_image_count(self) -> None:
        query = "Are these the same building?"
        no_images = interpret_request(self._request(query, []))
        one_image = interpret_request(self._request(query, [str(self.image_a)]))
        two_images = interpret_request(self._request(query, [str(self.image_a), str(self.image_b)]))
        self.assertEqual(no_images.family, "same_building_verification")
        self.assertEqual(one_image.family, "same_building_verification")
        self.assertEqual(two_images.family, "same_building_verification")

    def test_same_building_missing_images_requests_evidence(self) -> None:
        request = self._request("Are these the same building?", [str(self.image_a)])
        interpretation = interpret_request(request)
        decision = non_execution_decision(request, interpretation)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.action, "request_evidence")
        self.assertEqual(decision.reason_code, "missing_required_images")
        self.assertEqual(decision.required_images, 2)

    def test_same_building_extra_images_requests_evidence(self) -> None:
        request = self._request(
            "Are these the same building?",
            [str(self.image_a), str(self.image_b), str(self.image_c)],
        )
        interpretation = interpret_request(request)
        decision = non_execution_decision(request, interpretation)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.action, "request_evidence")
        self.assertEqual(decision.reason_code, "missing_required_images")
        self.assertEqual(decision.required_images, 2)

    def test_same_building_with_exactly_two_images_has_no_non_execution_decision(self) -> None:
        request = self._request("Are these the same building?", [str(self.image_a), str(self.image_b)])
        interpretation = interpret_request(request)
        decision = non_execution_decision(request, interpretation)
        self.assertIsNone(decision)

    def test_unsupported_family_abstains_without_backend(self) -> None:
        request = self._request("Identify the similarities and differences.", [str(self.image_a)])
        interpretation = interpret_request(request)
        decision = non_execution_decision(request, interpretation)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.family, "change_captioning")
        self.assertEqual(decision.action, "abstain")
        self.assertEqual(decision.reason_code, "capability_not_yet_supported")
        self.assertIsNone(decision.backend_id)

    def test_unknown_family_abstains_without_backend(self) -> None:
        request = self._request("Please do whatever makes sense.", [])
        interpretation, decision = classify_and_decide(request)
        self.assertEqual(interpretation.family, "unknown")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.action, "abstain")
        self.assertEqual(decision.reason_code, "unknown_query_family")
        self.assertIsNone(decision.backend_id)


if __name__ == "__main__":
    unittest.main()
