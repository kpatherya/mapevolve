from __future__ import annotations

from langchain_core.messages import AIMessage

from jev_router.classifier import JevQueryClassifier
from jev_router.pipeline import AdaptiveRouter
from jev_router.vlm import VLMExecutor, build_messages


class FakeChat:
    def __init__(self, key, log, fail=False, cost=None):
        self.key, self.log, self.fail, self.cost = key, log, fail, cost

    def invoke(self, messages):
        self.log.append((self.key, messages))
        if self.fail:
            raise TimeoutError("provider timeout")
        meta = {"finish_reason": "stop", "provider": "fake"}
        if self.cost is not None:
            meta["cost"] = self.cost
        return AIMessage(
            content=f"answer from {self.key}",
            usage_metadata={"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050},
            response_metadata=meta,
        )


def factory(log, failing=(), costs=None):
    costs = costs or {}

    def make(model, generation):
        assert generation["temperature"] == 0
        return FakeChat(model["key"], log, fail=model["key"] in failing, cost=costs.get(model["key"]))

    return make


def test_messages_have_labelled_images(sample_images):
    system, human = build_messages("SYS", "What changed?", sample_images)
    texts = [b["text"] for b in human.content if b["type"] == "text"]
    assert texts == ["Image A:", "Image B:", "Question: What changed?"]
    assert sum(b["type"] == "image_url" for b in human.content) == 2
    assert human.content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_one_image_query(sample_images):
    _, human = build_messages("SYS", "What building is this?", sample_images[:1])
    assert [b["type"] for b in human.content] == ["text", "image_url", "text"]


def test_provider_cost_and_usage(toy_registry, routing_config, sample_images):
    log = []
    ex = VLMExecutor(toy_registry, routing_config["generation"], factory(log, costs={"CHEAP": 0.0012}), "SYS")
    res = ex.run("CHEAP", "What changed?", sample_images)
    assert res.status == "ok" and res.cost_usd == 0.0012 and res.cost_source == "provider_reported"
    assert (res.input_tokens, res.output_tokens) == (1000, 50)
    assert res.latency_seconds is not None


def test_cost_estimated_when_provider_does_not_report(toy_registry, routing_config, sample_images):
    ex = VLMExecutor(toy_registry, routing_config["generation"], factory([]), "SYS")
    res = ex.run("MID", "What changed?", sample_images)
    assert res.cost_source == "estimated_from_registry_pricing"
    assert res.cost_usd == round(1000 * 0.5 / 1e6 + 50 * 2.0 / 1e6, 6)


def test_failover_and_identical_prompts(toy_registry, routing_config, sample_images):
    log = []
    ex = VLMExecutor(toy_registry, routing_config["generation"], factory(log, failing={"PREMIUM"}), "SYS")
    final, attempts = ex.run_with_failover(["PREMIUM", "MID"], "What changed?", sample_images)
    assert final.key == "MID"
    assert [a.status for a in attempts] == ["error", "ok"]
    assert "provider timeout" in attempts[0].error
    # Both models received byte-identical messages.
    assert log[0][1] == log[1][1]


def test_pipeline_end_to_end_with_mocks(toy_registry, routing_config, sample_images, tmp_path):
    class BrokenJev:
        def invoke(self, request, config=None):
            raise ConnectionError("no network")

    log = []
    router = AdaptiveRouter(
        classifier=JevQueryClassifier(decision_model=BrokenJev()),
        registry=toy_registry,
        executor=VLMExecutor(toy_registry, routing_config["generation"], factory(log), "SYS"),
        config=routing_config,
        trace_dir=tmp_path,
    )
    trace = router.answer("What changed between these two images?", sample_images, mode="economy",
                          budget_usd=1.0, compare=True)
    assert trace["classification"]["source"] == "heuristic_fallback"
    assert trace["result"]["status"] == "ok"
    assert {r["strategy"] for r in trace["comparison"]} == {"cheapest_eligible", "strongest", "jev_adaptive"}
    saved = list(tmp_path.glob("*.jsonl"))
    assert len(saved) == 1 and trace["trace_id"] in saved[0].read_text()


def test_openrouter_timeout_is_milliseconds(monkeypatch):
    from jev_router.vlm import default_chat_factory

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")  # client construction only; no call is made

    gen = {"temperature": 0, "max_output_tokens": 64, "timeout_seconds": 120, "max_retries": 1}
    chat = default_chat_factory({"backend": "openrouter", "model_id": "google/gemma-3-27b-it"}, gen)
    assert chat.request_timeout == 120_000
