import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from mapevolve.backends import OpenRouterBackend
from mapevolve.contracts import RoutingRequest
from mapevolve.jev_classifier import JevQueryClassifier, build_state
from mapevolve.pipeline import run_pipeline


def _response(label="cross_view_matching", confidence=0.9, diff=None):
    diff = diff if diff is not None else {0: 0.1, 1: 0.2, 2: 0.7}
    return NS(
        choices={"query_type": NS(choice=label, probabilities={label: confidence}, confidence=confidence)},
        scores={"difficulty": NS(probabilities=diff, confidence=0.7)},
        model="jev-test",
        request_id="req-1",
    )


class FakeJev:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def invoke(self, payload, config=None):
        self.calls.append(payload)
        if self.error:
            raise self.error
        return self.response


class JevClassifierTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.images = []
        for name in ("a.jpg", "b.jpg"):
            path = Path(self.dir.name, name)
            path.write_bytes(b"\xff\xd8x")
            self.images.append(str(path))
        self.request = RoutingRequest(request_id="r1", query="Are these the same building?", images=self.images)

    def _classifier(self, jev):
        return JevQueryClassifier(jev, questions_factory=lambda: {})

    def test_jev_result_maps_taxonomy_and_difficulty(self):
        result = self._classifier(FakeJev(_response()))(self.request)
        self.assertEqual(result.source, "jev")
        self.assertEqual(result.interpretation.family, "same_building_verification")
        self.assertEqual(result.difficulty, "hard")
        self.assertEqual(result.jev_request_id, "req-1")

    def test_landmark_identification_maps_to_unknown(self):
        result = self._classifier(FakeJev(_response("landmark_identification")))(self.request)
        self.assertEqual(result.interpretation.family, "unknown")

    def test_jev_state_has_no_image_paths_or_labels(self):
        jev = FakeJev(_response())
        self._classifier(jev)(self.request)
        state = jev.calls[0]["state"]
        self.assertEqual(state, {"question": self.request.query, "num_images": 2})

    def test_leak_guard_rejects_label_metadata(self):
        with self.assertRaises(ValueError):
            build_state(self.request, {"ground_truth": "yes"})

    def test_error_falls_back_to_rules_with_reason(self):
        result = self._classifier(FakeJev(error=RuntimeError("down")))(self.request)
        self.assertEqual(result.source, "rules_fallback")
        self.assertIn("down", result.fallback_reason)
        self.assertEqual(result.interpretation.family, "same_building_verification")

    def test_invalid_label_falls_back(self):
        result = self._classifier(FakeJev(_response("bogus")))(self.request)
        self.assertEqual(result.source, "rules_fallback")

    def test_missing_key_falls_back_without_sdk(self):
        import os

        saved = os.environ.pop("TYPESAFE_API_KEY", None)
        try:
            result = JevQueryClassifier()(self.request)
        finally:
            if saved is not None:
                os.environ["TYPESAFE_API_KEY"] = saved
        self.assertEqual(result.source, "rules_fallback")
        self.assertIn("TYPESAFE_API_KEY", result.fallback_reason)

    def test_pipeline_records_classification_and_uses_jev_family(self):
        trace = run_pipeline(
            RoutingRequest(request_id="r2", query="zzz no keyword", images=self.images),
            backend_registry={},
            dry_run=True,
            payload_builders={
                "openai/gpt-4.1-mini": lambda r: {"x": 1},
                "anthropic/claude-3.5-sonnet": lambda r: {"x": 1},
            },
            classifier=self._classifier(FakeJev(_response())),
        )
        data = trace.to_dict()
        self.assertEqual(data["classification"]["source"], "jev")
        self.assertEqual(data["decision"]["action"], "execute")
        self.assertGreater(data["dry_run_payload_bytes"], 0)
        json.dumps(data)


class BackendPayloadTests(unittest.TestCase):
    def test_prompt_contains_cross_view_guidance_and_json_contract(self):
        from mapevolve.backends import OPENROUTER_SYSTEM_PROMPT as p

        self.assertIn("Look-alike", p)
        self.assertIn("answer, evidence", p)

    def test_allowed_model_ids_extends_backend_models(self):
        backend = OpenRouterBackend(
            model_id="new/model", api_key="k", input_cost_usd_per_1m_tokens=1.0, image_cost_usd=0.0,
            allowed_model_ids=frozenset({"new/model"}),
        )
        self.assertEqual(backend.backend_id, "new/model")
        with self.assertRaises(ValueError):
            OpenRouterBackend(model_id="new/model", api_key="k", input_cost_usd_per_1m_tokens=1.0, image_cost_usd=0.0)


if __name__ == "__main__":
    unittest.main()
