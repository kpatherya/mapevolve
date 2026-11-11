"""Model registry access: capabilities, prices, and shrunk task scores."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config" / "model_registry.json"
ROUTING_PATH = ROOT / "config" / "routing.yaml"


def load_routing_config(path: Path = ROUTING_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass
class ScoreEstimate:
    quality: Optional[float]            # normalized 0..1
    unsupported_rate: Optional[float]
    source: str                         # e.g. "temporal_comparison/hard (n=2, shrunk k=2)"


class ModelRegistry:
    def __init__(self, data: dict[str, Any], shrinkage_k: float = 2.0, defaults: Optional[dict] = None):
        self.data = data
        self.shrinkage_k = shrinkage_k
        self.defaults = defaults or {}
        self.models = {m["key"]: m for m in data["models"]}

    @classmethod
    def load(cls, path: Path = REGISTRY_PATH, routing: Optional[dict[str, Any]] = None) -> "ModelRegistry":
        routing = routing or load_routing_config()
        reg_cfg = routing.get("registry", {})
        return cls(json.loads(path.read_text(encoding="utf-8")), reg_cfg.get("shrinkage_k", 2), reg_cfg)

    def with_metrics(self, metrics: dict[str, dict[str, Any]]) -> "ModelRegistry":
        """Copy of the registry with evaluation metrics replaced (leave-one-out eval)."""
        data = copy.deepcopy(self.data)
        for model in data["models"]:
            m = metrics.get(model["key"])
            if m is None:
                model["evaluation"] = None
                continue
            model["evaluation"] = {
                "general": m["general"],
                "unsupported_claim_rate": {"rate": m["general"]["unsupported_claim_rate"], "n": m["general"]["n"]},
                "task_scores": m["task_scores"],
                "score_sources": m["score_sources"],
            }
            model["latency"] = {**(model.get("latency") or {}), **m["latency"]}
        return ModelRegistry(data, self.shrinkage_k, self.defaults)

    # ------------------------------------------------------------------ #
    def keys(self) -> list[str]:
        return list(self.models)

    def get(self, key: str) -> dict[str, Any]:
        return self.models[key]

    def is_listed(self, key: str) -> bool:
        return self.models[key]["availability"]["status"] in {"listed", "expiring", "unknown"}

    def supports_images(self, key: str, n_images: int) -> bool:
        info = self.models[key]["image_input"]
        return bool(info.get("supported")) and n_images <= int(info.get("max_images", 1))

    def general_quality(self, key: str) -> Optional[float]:
        ev = self.models[key].get("evaluation")
        return ev["general"]["normalized"] if ev else None

    def score(self, key: str, task: str, difficulty: str) -> ScoreEstimate:
        """Task x difficulty score shrunk toward the general score.

        Fallback order: task/difficulty cell -> task overall -> general.
        """
        ev = self.models[key].get("evaluation")
        if not ev:
            return ScoreEstimate(None, None, "no evaluation data")
        general = ev["general"]
        g_q, g_u = general["normalized"], general["unsupported_claim_rate"]
        k = self.shrinkage_k
        task_scores = ev.get("task_scores", {}).get(task)
        if task_scores:
            cell = task_scores["by_difficulty"].get(difficulty)
            level = f"{task}/{difficulty}"
            if cell is None:
                cell, level = task_scores["overall"], f"{task}/all"
            n = cell["n"]
            q = (n * cell["normalized"] + k * g_q) / (n + k)
            u = (n * cell["unsupported_claim_rate"] + k * g_u) / (n + k)
            return ScoreEstimate(q, u, f"{level} (n={n}, shrunk k={k:g})")
        return ScoreEstimate(g_q, g_u, f"general fallback; no {task} data (n={general['n']})")

    def estimate_cost(self, key: str, n_images: int) -> tuple[Optional[float], str]:
        """Per-query cost estimate: benchmark token counts x current prices.

        Falls back to the historical observed mean cost when no current price
        exists (e.g. the model is no longer listed).
        """
        model = self.models[key]
        pricing = model.get("pricing")
        tokens = model.get("tokens") or {}
        per_image = self.defaults.get("default_tokens_per_image", 1100)
        if tokens.get("mean_input_tokens_per_query"):
            images_in_bench = tokens.get("images_per_query", 2) or 2
            in_tok = tokens["mean_input_tokens_per_query"] * max(n_images, 1) / images_in_bench
            out_tok = tokens.get("mean_output_tokens_per_query") or self.defaults.get("default_output_tokens", 300)
            token_source = "benchmark tokens"
        else:
            in_tok = per_image * max(n_images, 1) + 300
            out_tok = self.defaults.get("default_output_tokens", 300)
            token_source = "default token assumption"
        if pricing and pricing.get("input_usd_per_mtok") is not None:
            cost = in_tok * pricing["input_usd_per_mtok"] / 1e6 + out_tok * (pricing.get("output_usd_per_mtok") or 0) / 1e6
            if pricing.get("image_usd_per_image"):
                cost += pricing["image_usd_per_image"] * n_images
            return cost, f"current {pricing['source']} price x {token_source}"
        observed = (model.get("observed_cost") or {}).get("mean_usd_per_query")
        if observed is not None:
            return observed * max(n_images, 1) / 2, "historical observed cost (no current price)"
        return None, "unknown"

    def latency(self, key: str) -> Optional[float]:
        return (self.models[key].get("latency") or {}).get("mean_seconds")
