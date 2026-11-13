from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from mapevolve.backends import FakeBackend, FakeBackendCase, OpenRouterBackend
from mapevolve.cli import main as cli_main
from mapevolve.contracts import RoutingRequest
from mapevolve.pipeline import run_pipeline


class PipelineAndCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.image_a = Path(self.tmpdir.name) / "a.png"
        self.image_b = Path(self.tmpdir.name) / "b.png"
        png_bytes = b"\x89PNG\r\n\x1a\n"
        self.image_a.write_bytes(png_bytes)
        self.image_b.write_bytes(png_bytes)

    def _request(self, request_id: str, query: str, image_count: int = 2) -> RoutingRequest:
        images = []
        if image_count >= 1:
            images.append(str(self.image_a))
        if image_count >= 2:
            images.append(str(self.image_b))
        return RoutingRequest(
            request_id=request_id,
            query=query,
            images=images,
        )

    def _fake_registry(self) -> dict[str, FakeBackend]:
        return {
            "openai/gpt-4.1-mini": FakeBackend(
                backend_id="openai/gpt-4.1-mini",
                scripted_cases={
                    "req-positive": FakeBackendCase(
                        status="success",
                        answer="true",
                        evidence=("window grid matches",),
                        raw_content='{"answer":"true","evidence":["window grid matches"]}',
                    ),
                    "req-negative": FakeBackendCase(
                        status="success",
                        answer="false",
                        evidence=("roofline differs",),
                        raw_content='{"answer":"false","evidence":["roofline differs"]}',
                    ),
                    "req-unknown": FakeBackendCase(
                        status="success",
                        answer="unknown",
                        evidence=("occluded view",),
                        raw_content='{"answer":"unknown","evidence":["occluded view"]}',
                    ),
                },
            )
        }

    def test_fake_pipeline_preserves_true_false_unknown_outputs(self) -> None:
        registry = self._fake_registry()
        cases = [
            ("req-positive", "Are these the same building?", "true"),
            ("req-negative", "Are these the same building?", "false"),
            ("req-unknown", "Are these the same building?", "unknown"),
        ]
        for request_id, query, expected_answer in cases:
            trace = run_pipeline(self._request(request_id, query), backend_registry=registry)
            self.assertEqual(trace.decision.action, "execute")
            self.assertIsNotNone(trace.backend_result)
            assert trace.backend_result is not None
            self.assertEqual(trace.backend_result.answer, expected_answer)

    def test_request_evidence_path_has_no_execution_result(self) -> None:
        trace = run_pipeline(
            self._request("req-missing", "Are these the same building?", image_count=1),
            backend_registry=self._fake_registry(),
        )
        self.assertEqual(trace.decision.action, "request_evidence")
        self.assertIsNone(trace.backend_result)

    def test_unsupported_family_has_no_execution_result(self) -> None:
        trace = run_pipeline(
            self._request("req-unsupported", "Identify the similarities and differences.", image_count=2),
            backend_registry=self._fake_registry(),
        )
        self.assertEqual(trace.decision.action, "abstain")
        self.assertIsNone(trace.backend_result)

    def test_dry_run_builds_payload_without_network_call(self) -> None:
        transport_called = {"value": False}

        def transport(*_: Any) -> dict[str, Any]:
            transport_called["value"] = True
            raise AssertionError("transport should not be called during dry-run")

        backend = OpenRouterBackend(
            model_id="openai/gpt-4.1-mini",
            api_key="dry-run-key",
            input_cost_usd_per_1m_tokens=0.4,
            image_cost_usd=0.003,
            transport=transport,
        )
        trace = run_pipeline(
            self._request("req-dry-run", "Are these the same building?", image_count=2),
            backend_registry={"openai/gpt-4.1-mini": backend},
            dry_run=True,
        )

        self.assertFalse(transport_called["value"])
        self.assertEqual(trace.decision.action, "execute")
        self.assertIsNone(trace.backend_result)
        self.assertIsNotNone(trace.dry_run_payload)
        payload_dump = json.dumps(trace.dry_run_payload)
        self.assertIn("data:image/png;base64,<redacted>", payload_dump)
        self.assertNotIn("iVBORw0KGgo", payload_dump)

    def test_cli_dry_run_records_selected_profile_and_backend(self) -> None:
        output_path = Path(self.tmpdir.name) / "trace.json"
        exit_code = cli_main(
            [
                "--request-id",
                "req-cli",
                "--query",
                "Are these the same building?",
                "--image",
                str(self.image_a),
                "--image",
                str(self.image_b),
                "--dry-run",
                "--backend-mode",
                "fake",
                "--output",
                str(output_path),
            ]
        )
        self.assertEqual(exit_code, 0)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["decision"]["action"], "execute")
        self.assertEqual(payload["decision"]["backend_id"], "openai/gpt-4.1-mini")
        self.assertEqual(payload["policy"]["profile_version"], "mvp_v1")


if __name__ == "__main__":
    unittest.main()
