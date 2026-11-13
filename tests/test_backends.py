from __future__ import annotations

import json
import tempfile
import urllib.error
import unittest
from pathlib import Path
from typing import Any

from mapevolve.backends import FakeBackend, FakeBackendCase, OpenRouterBackend, SUPPORTED_OPENROUTER_MODELS
from mapevolve.contracts import RoutingRequest


class BackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.image_dir_a = Path(self.tmpdir.name) / "building_alpha"
        self.image_dir_b = Path(self.tmpdir.name) / "building_beta"
        self.image_dir_a.mkdir(parents=True, exist_ok=True)
        self.image_dir_b.mkdir(parents=True, exist_ok=True)
        self.image_a = self.image_dir_a / "view_a.png"
        self.image_b = self.image_dir_b / "view_b.png"
        png_bytes = b"\x89PNG\r\n\x1a\n"
        self.image_a.write_bytes(png_bytes)
        self.image_b.write_bytes(png_bytes)

    def _request(self, request_id: str = "req") -> RoutingRequest:
        return RoutingRequest(
            request_id=request_id,
            query="Are these the same building?",
            images=[str(self.image_a), str(self.image_b)],
        )

    def test_fake_backend_supports_true_false_unknown(self) -> None:
        fake = FakeBackend(
            scripted_cases={
                "r-true": FakeBackendCase(
                    status="success",
                    answer="true",
                    evidence=("window pattern matches",),
                    raw_content='{"answer":"true","evidence":["window pattern matches"]}',
                ),
                "r-false": FakeBackendCase(
                    status="success",
                    answer="false",
                    evidence=("roofline differs",),
                    raw_content='{"answer":"false","evidence":["roofline differs"]}',
                ),
                "r-unknown": FakeBackendCase(
                    status="success",
                    answer="unknown",
                    evidence=("occlusion prevents certainty",),
                    raw_content='{"answer":"unknown","evidence":["occlusion prevents certainty"]}',
                ),
            }
        )
        self.assertEqual(fake.execute(self._request("r-true")).answer, "true")
        self.assertEqual(fake.execute(self._request("r-false")).answer, "false")
        self.assertEqual(fake.execute(self._request("r-unknown")).answer, "unknown")

    def test_fake_backend_supports_failure_states(self) -> None:
        fake = FakeBackend(
            scripted_cases={
                "r-invalid": FakeBackendCase(
                    status="invalid_output",
                    answer=None,
                    error="malformed backend output",
                    raw_content="not-json",
                ),
                "r-timeout": FakeBackendCase(status="timeout", answer=None, error="timed out"),
                "r-error": FakeBackendCase(status="error", answer=None, error="network failure"),
            }
        )
        self.assertEqual(fake.execute(self._request("r-invalid")).status, "invalid_output")
        self.assertEqual(fake.execute(self._request("r-timeout")).status, "timeout")
        self.assertEqual(fake.execute(self._request("r-error")).status, "error")

    def test_openrouter_payload_includes_two_images_without_path_labels(self) -> None:
        backend = OpenRouterBackend(
            model_id="openai/gpt-4.1-mini",
            api_key="test-key",
            input_cost_usd_per_1m_tokens=0.4,
            image_cost_usd=0.003,
            transport=lambda *_: {},
        )
        payload = backend.build_payload(self._request("req-payload"))
        content = payload["messages"][1]["content"]
        image_items = [item for item in content if item["type"] == "image_url"]
        self.assertEqual(len(image_items), 2)
        self.assertTrue(all(item["image_url"]["url"].startswith("data:image/png;base64,") for item in image_items))

        payload_blob = json.dumps(payload)
        self.assertNotIn("building_alpha", payload_blob)
        self.assertNotIn("building_beta", payload_blob)
        self.assertNotIn("view_a.png", payload_blob)
        self.assertNotIn("view_b.png", payload_blob)

    def test_openrouter_execute_success_preserves_unknown_and_estimates_cost(self) -> None:
        def transport(*_: Any) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"answer":"unknown","evidence":["facade aligns but occluded"]}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 500},
            }

        backend = OpenRouterBackend(
            model_id="openai/gpt-4.1-mini",
            api_key="test-key",
            input_cost_usd_per_1m_tokens=2.0,
            image_cost_usd=0.01,
            transport=transport,
        )
        result = backend.execute(self._request("req-ok"))
        self.assertEqual(result.status, "success")
        self.assertEqual(result.answer, "unknown")
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.estimated_cost_usd or 0.0, 0.021, places=6)

    def test_openrouter_maps_malformed_output_to_invalid_output(self) -> None:
        def transport(*_: Any) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "not-json"}}], "usage": {"prompt_tokens": 100}}

        backend = OpenRouterBackend(
            model_id="openai/gpt-4.1-mini",
            api_key="test-key",
            input_cost_usd_per_1m_tokens=1.0,
            image_cost_usd=0.001,
            transport=transport,
        )
        result = backend.execute(self._request("req-invalid"))
        self.assertEqual(result.status, "invalid_output")
        self.assertIsNone(result.answer)
        self.assertIsNotNone(result.error)

    def test_openrouter_http_error_preserves_redacted_body(self) -> None:
        import io

        def transport(*_: Any) -> dict[str, Any]:
            raise urllib.error.HTTPError(
                "http://x", 400, "Bad Request", {}, io.BytesIO(b'{"error":"bad model test-key"}')
            )

        backend = OpenRouterBackend(
            model_id="openai/gpt-4.1-mini",
            api_key="test-key",
            input_cost_usd_per_1m_tokens=1.0,
            image_cost_usd=0.001,
            transport=transport,
        )
        result = backend.execute(self._request("req-http"))
        self.assertEqual(result.status, "error")
        self.assertIn("bad model", result.error or "")
        self.assertNotIn("test-key", result.error or "")

    def test_openrouter_maps_timeout(self) -> None:
        def transport(*_: Any) -> dict[str, Any]:
            raise TimeoutError("request timed out")

        backend = OpenRouterBackend(
            model_id="openai/gpt-4.1-mini",
            api_key="test-key",
            input_cost_usd_per_1m_tokens=1.0,
            image_cost_usd=0.001,
            transport=transport,
        )
        result = backend.execute(self._request("req-timeout"))
        self.assertEqual(result.status, "timeout")
        self.assertIsNone(result.answer)
        self.assertIsNotNone(result.error)

    def test_openrouter_maps_transport_errors(self) -> None:
        def transport(*_: Any) -> dict[str, Any]:
            raise urllib.error.URLError("connection failed")

        backend = OpenRouterBackend(
            model_id="openai/gpt-4.1-mini",
            api_key="test-key",
            input_cost_usd_per_1m_tokens=1.0,
            image_cost_usd=0.001,
            transport=transport,
        )
        result = backend.execute(self._request("req-error"))
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.answer)
        self.assertIsNotNone(result.error)

    def test_openrouter_supported_models_expose_multiple_tiers(self) -> None:
        self.assertIn("openai/gpt-4.1-mini", SUPPORTED_OPENROUTER_MODELS)
        self.assertIn("anthropic/claude-3.5-sonnet", SUPPORTED_OPENROUTER_MODELS)
        self.assertGreaterEqual(len(SUPPORTED_OPENROUTER_MODELS), 2)


if __name__ == "__main__":
    unittest.main()
