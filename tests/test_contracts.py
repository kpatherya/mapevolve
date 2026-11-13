from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from mapevolve.contracts import (
    ROUTING_SCHEMA_VERSION,
    BackendResult,
    ContractValidationError,
    QueryInterpretation,
    RoutingDecision,
    RoutingRequest,
)


class RoutingRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.image_path = Path(self.tmpdir.name) / "sample.png"
        self.image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    def test_valid_request_round_trip(self) -> None:
        request = RoutingRequest(
            request_id="req-1",
            query="Are these the same building?",
            images=[str(self.image_path)],
            schema_version=ROUTING_SCHEMA_VERSION,
        )
        encoded = request.to_json()
        decoded = RoutingRequest.from_json(encoded)
        self.assertEqual(request, decoded)

    def test_invalid_image_extension_rejected(self) -> None:
        text_file = Path(self.tmpdir.name) / "bad.txt"
        text_file.write_text("not an image", encoding="utf-8")
        with self.assertRaises(ContractValidationError):
            RoutingRequest(
                request_id="req-2",
                query="q",
                images=[str(text_file)],
                schema_version=ROUTING_SCHEMA_VERSION,
            )

    def test_missing_schema_version_rejected(self) -> None:
        payload = {"request_id": "req-3", "query": "q", "images": [str(self.image_path)]}
        with self.assertRaises(ContractValidationError):
            RoutingRequest.from_dict(payload)

    def test_unknown_fields_rejected(self) -> None:
        payload = {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "request_id": "req-3",
            "query": "q",
            "images": [str(self.image_path)],
            "extra": "unexpected",
        }
        with self.assertRaises(ContractValidationError):
            RoutingRequest.from_dict(payload)

    def test_missing_required_fields_rejected(self) -> None:
        payload = {"schema_version": ROUTING_SCHEMA_VERSION, "request_id": "req-3", "query": "q"}
        with self.assertRaises(ContractValidationError):
            RoutingRequest.from_dict(payload)

    def test_wrong_field_type_rejected(self) -> None:
        payload = {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "request_id": "req-3",
            "query": "q",
            "images": "not-a-list",
        }
        with self.assertRaises(ContractValidationError):
            RoutingRequest.from_dict(payload)


class QueryInterpretationTests(unittest.TestCase):
    def test_valid_round_trip(self) -> None:
        interpretation = QueryInterpretation(
            request_id="req-1",
            family="same_building_verification",
            rule_id="same_building_v1",
            confidence=0.9,
            schema_version=ROUTING_SCHEMA_VERSION,
        )
        self.assertEqual(interpretation, QueryInterpretation.from_json(interpretation.to_json()))

    def test_unknown_family_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            QueryInterpretation(
                request_id="req-1",
                family="made_up",
                rule_id="r",
                confidence=None,
                schema_version=ROUTING_SCHEMA_VERSION,
            )

    def test_all_families_supported(self) -> None:
        families = [
            "same_building_verification",
            "change_captioning",
            "evidence_backed_investigation",
            "proximity_computation",
            "unknown",
        ]
        for family in families:
            interpretation = QueryInterpretation(
                request_id="req-all",
                family=family,
                rule_id="rule",
                confidence=0.1,
                schema_version=ROUTING_SCHEMA_VERSION,
            )
            self.assertEqual(interpretation.family, family)

    def test_unknown_fields_rejected(self) -> None:
        payload = {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "request_id": "req-1",
            "family": "unknown",
            "rule_id": "rule",
            "confidence": 0.5,
            "extra": "unexpected",
        }
        with self.assertRaises(ContractValidationError):
            QueryInterpretation.from_dict(payload)

    def test_missing_required_fields_rejected(self) -> None:
        payload = {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "request_id": "req-1",
            "rule_id": "rule",
        }
        with self.assertRaises(ContractValidationError):
            QueryInterpretation.from_dict(payload)

    def test_wrong_field_type_rejected(self) -> None:
        payload = {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "request_id": "req-1",
            "family": "unknown",
            "rule_id": "rule",
            "confidence": "high",
        }
        with self.assertRaises(ContractValidationError):
            QueryInterpretation.from_dict(payload)


class RoutingDecisionTests(unittest.TestCase):
    def test_execute_action_requires_backend(self) -> None:
        with self.assertRaises(ContractValidationError):
            RoutingDecision(
                request_id="req-4",
                family="same_building_verification",
                action="execute",
                reason_code="ready",
                backend_id=None,
                required_images=None,
                schema_version=ROUTING_SCHEMA_VERSION,
            )

    def test_request_evidence_requires_required_images(self) -> None:
        with self.assertRaises(ContractValidationError):
            RoutingDecision(
                request_id="req-5",
                family="same_building_verification",
                action="request_evidence",
                reason_code="missing_required_images",
                backend_id=None,
                required_images=None,
                schema_version=ROUTING_SCHEMA_VERSION,
            )

    def test_abstain_round_trip(self) -> None:
        decision = RoutingDecision(
            request_id="req-6",
            family="unknown",
            action="abstain",
            reason_code="unknown_query_family",
            backend_id=None,
            required_images=None,
            schema_version=ROUTING_SCHEMA_VERSION,
        )
        self.assertEqual(decision, RoutingDecision.from_dict(json.loads(decision.to_json())))

    def test_unknown_fields_rejected(self) -> None:
        payload = {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "request_id": "req-6",
            "family": "unknown",
            "action": "abstain",
            "reason_code": "unknown_query_family",
            "backend_id": None,
            "required_images": None,
            "extra": "unexpected",
        }
        with self.assertRaises(ContractValidationError):
            RoutingDecision.from_dict(payload)

    def test_missing_required_fields_rejected(self) -> None:
        payload = {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "request_id": "req-6",
            "family": "unknown",
            "action": "abstain",
        }
        with self.assertRaises(ContractValidationError):
            RoutingDecision.from_dict(payload)

    def test_wrong_field_type_rejected(self) -> None:
        payload = {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "request_id": "req-6",
            "family": "same_building_verification",
            "action": "request_evidence",
            "reason_code": "missing_required_images",
            "required_images": "2",
        }
        with self.assertRaises(ContractValidationError):
            RoutingDecision.from_dict(payload)


class BackendResultTests(unittest.TestCase):
    def _base_payload(self) -> dict[str, object]:
        return {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "request_id": "req-b",
            "backend_id": "fake-backend",
            "status": "success",
            "answer": "true",
            "evidence": ["visible window grid matches"],
            "latency_ms": 5.0,
            "estimated_cost_usd": 0.001,
            "raw_content": '{"answer":"true"}',
            "error": None,
        }

    def test_success_round_trip_preserves_unknown_answer(self) -> None:
        result = BackendResult(
            request_id="req-7",
            backend_id="fake-backend",
            status="success",
            answer="unknown",
            evidence=["facade shape matches in both images"],
            latency_ms=21.4,
            estimated_cost_usd=0.0005,
            raw_content='{"answer":"unknown"}',
            error=None,
            schema_version=ROUTING_SCHEMA_VERSION,
        )
        self.assertEqual(result, BackendResult.from_json(result.to_json()))

    def test_invalid_answer_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            BackendResult(
                request_id="req-8",
                backend_id="fake-backend",
                status="success",
                answer="maybe",
                evidence=[],
                latency_ms=1.0,
                estimated_cost_usd=0.0,
                raw_content='{"answer":"maybe"}',
                error=None,
                schema_version=ROUTING_SCHEMA_VERSION,
            )

    def test_negative_latency_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            BackendResult(
                request_id="req-9",
                backend_id="fake-backend",
                status="error",
                answer=None,
                evidence=[],
                latency_ms=-1.0,
                estimated_cost_usd=None,
                raw_content=None,
                error="timeout",
                schema_version=ROUTING_SCHEMA_VERSION,
            )

    def test_raw_content_with_base64_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            BackendResult(
                request_id="req-10",
                backend_id="fake-backend",
                status="invalid_output",
                answer=None,
                evidence=[],
                latency_ms=1.0,
                estimated_cost_usd=None,
                raw_content="data:image/png;base64,abcd",
                error="bad schema",
                schema_version=ROUTING_SCHEMA_VERSION,
            )

    def test_unknown_fields_rejected(self) -> None:
        payload = self._base_payload()
        payload["extra"] = "unexpected"
        with self.assertRaises(ContractValidationError):
            BackendResult.from_dict(payload)

    def test_missing_required_fields_rejected(self) -> None:
        payload = self._base_payload()
        del payload["status"]
        with self.assertRaises(ContractValidationError):
            BackendResult.from_dict(payload)

    def test_wrong_field_type_rejected(self) -> None:
        payload = self._base_payload()
        payload["evidence"] = "not-a-list"
        with self.assertRaises(ContractValidationError):
            BackendResult.from_dict(payload)

    def test_non_success_statuses_require_error(self) -> None:
        for status in ("invalid_output", "timeout", "error"):
            payload = self._base_payload()
            payload["status"] = status
            payload["answer"] = None
            payload["error"] = None
            with self.assertRaises(ContractValidationError):
                BackendResult.from_dict(payload)

    def test_non_success_statuses_reject_answer(self) -> None:
        for status in ("invalid_output", "timeout", "error"):
            payload = self._base_payload()
            payload["status"] = status
            payload["answer"] = "true"
            payload["error"] = "failed"
            with self.assertRaises(ContractValidationError):
                BackendResult.from_dict(payload)

    def test_non_finite_latency_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            payload = self._base_payload()
            payload["latency_ms"] = value
            with self.assertRaises(ContractValidationError):
                BackendResult.from_dict(payload)

    def test_non_finite_cost_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            payload = self._base_payload()
            payload["estimated_cost_usd"] = value
            with self.assertRaises(ContractValidationError):
                BackendResult.from_dict(payload)

    def test_negative_cost_rejected(self) -> None:
        payload = self._base_payload()
        payload["estimated_cost_usd"] = -0.001
        with self.assertRaises(ContractValidationError):
            BackendResult.from_dict(payload)

    def test_empty_evidence_string_rejected(self) -> None:
        payload = self._base_payload()
        payload["evidence"] = [""]
        with self.assertRaises(ContractValidationError):
            BackendResult.from_dict(payload)

    def test_success_requires_answer_and_no_error(self) -> None:
        no_answer = self._base_payload()
        no_answer["answer"] = None
        with self.assertRaises(ContractValidationError):
            BackendResult.from_dict(no_answer)

        has_error = self._base_payload()
        has_error["error"] = "should not be present"
        with self.assertRaises(ContractValidationError):
            BackendResult.from_dict(has_error)


if __name__ == "__main__":
    unittest.main()
