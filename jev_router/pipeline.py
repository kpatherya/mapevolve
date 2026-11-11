"""End-to-end orchestration.

    classify (Jev) -> choose action -> route model -> answer -> check answer
                   -> escalate once if risky (cascade) -> trace
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from jev_router.actions import decide_action, decline_message
from jev_router.answer_check import AnswerChecker
from jev_router.classifier import JevQueryClassifier
from jev_router.image_meta import ImageSource
from jev_router.policy import RoutingPolicy
from jev_router.registry import ROOT, ModelRegistry, load_routing_config
from jev_router.schemas import QueryClassification, RoutingDecision, VLMResult
from jev_router.vlm import VLMExecutor

TRACE_DIR = ROOT / "runs" / "routing_traces"


def _image_hash(img: ImageSource) -> str:
    data = Path(img).read_bytes() if isinstance(img, (str, Path)) else img
    return hashlib.sha256(data).hexdigest()


class AdaptiveRouter:
    def __init__(
        self,
        classifier: Optional[JevQueryClassifier] = None,
        registry: Optional[ModelRegistry] = None,
        executor: Optional[VLMExecutor] = None,
        config: Optional[dict[str, Any]] = None,
        trace_dir: Optional[Path] = TRACE_DIR,
        checker: Optional[AnswerChecker] = None,
        use_quality_model: Optional[bool] = None,
    ) -> None:
        self.config = config or load_routing_config()
        self.registry = registry or ModelRegistry.load(routing=self.config)
        jev_cfg = self.config.get("jev", {})
        self.classifier = classifier or JevQueryClassifier(
            model=jev_cfg.get("model", "jev-latest"),
            timeout=jev_cfg.get("timeout_seconds", 20),
            image_aware=jev_cfg.get("image_aware", False),
        )
        self.policy = RoutingPolicy(self.registry, self.config, use_quality_model=use_quality_model)
        self.executor = executor or VLMExecutor(self.registry, self.config["generation"])
        self.checker = checker or AnswerChecker(timeout=jev_cfg.get("timeout_seconds", 20))
        self.trace_dir = trace_dir

    def classify_and_route(
        self,
        question: str,
        images: Sequence[ImageSource],
        mode: str,
        budget_usd: Optional[float] = None,
        image_aware: Optional[bool] = None,
        metadata: Optional[dict[str, Any]] = None,
        evidence: Optional[str] = None,
    ) -> tuple[QueryClassification, Any, RoutingDecision]:
        classification = self.classifier.classify(question, images, metadata, image_aware=image_aware, evidence=evidence)
        action = decide_action(classification, self.config, evidence)
        decision = self.policy.route(classification, len(images), mode, budget_usd)
        return classification, action, decision

    def answer(
        self,
        question: str,
        images: Sequence[ImageSource],
        mode: str = "balanced",
        budget_usd: Optional[float] = None,
        image_aware: Optional[bool] = None,
        execute: bool = True,
        compare: bool = False,
        evidence: Optional[str] = None,
        cascade: Optional[bool] = None,
    ) -> dict[str, Any]:
        classification, action, decision = self.classify_and_route(
            question, images, mode, budget_usd, image_aware, evidence=evidence)
        ccfg = self.config.get("cascade", {})
        cascade = ccfg.get("enabled", False) if cascade is None else cascade
        run_kwargs = dict(task=classification.query_type.label, evidence=evidence, note=action.prompt_note)
        trace: dict[str, Any] = {
            "trace_id": uuid.uuid4().hex[:12],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": question,
            "supplied_evidence": evidence,
            "n_images": len(images),
            "image_sha256": [_image_hash(i) for i in images],
            "mode": mode,
            "classification": classification.to_dict(),
            "action": action.to_dict(),
            "routing": decision.to_dict(),
            "quality_model": self.policy.quality_model_status,
            "result": None,
            "attempts": [],
            "answer_check": None,
            "cascade": None,
            "comparison": None,
        }
        if execute and action.action == "decline":
            trace["result"] = VLMResult(key="none", model_id="none", backend="none", status="declined",
                                        answer=decline_message(classification), latency_seconds=0.0,
                                        cost_usd=0.0, cost_source="no_call").to_dict()
        elif execute and decision.selected_key:
            chain = [decision.selected_key, *decision.fallback_keys]
            final, attempts = self.executor.run_with_failover(chain, question, images, **run_kwargs)
            trace["attempts"] = [a.to_dict() for a in attempts]
            if final and final.key != decision.selected_key:
                decision.warnings.append(f"{decision.selected_key} failed; answered by fallback {final.key}.")
                trace["routing"] = decision.to_dict()
            if final and final.answer:
                check = self.checker.check(final.answer, question, images, classification, evidence)
                trace["answer_check"] = check.to_dict()
                if cascade and check.risk >= ccfg.get("risk_threshold", 0.4):
                    final = self._escalate(trace, final, check, classification, question, images, budget_usd,
                                           evidence, run_kwargs)
            trace["result"] = final.to_dict() if final else None
        if execute and compare and action.action != "decline":
            trace["comparison"] = self.compare(question, images, trace, run_kwargs)
        self.save_trace(trace)
        return trace

    def _escalate(self, trace, first: VLMResult, first_check, classification, question, images, budget_usd,
                  evidence, run_kwargs) -> VLMResult:
        ccfg = self.config.get("cascade", {})
        esc = self.policy.route(classification, len(images), ccfg.get("escalate_mode", "accuracy"), budget_usd)
        order = [k for k in [esc.selected_key, *esc.fallback_keys] if k and k != first.key]
        info: dict[str, Any] = {"triggered": True, "first_key": first.key, "first_risk": first_check.risk,
                                "threshold": ccfg.get("risk_threshold", 0.4), "escalated_key": None}
        if not order:
            info["note"] = "No different eligible model to escalate to."
            trace["cascade"] = info
            return first
        second = self.executor.run(order[0], question, images, **run_kwargs)
        trace["attempts"].append(second.to_dict())
        info["escalated_key"] = second.key
        if second.status != "ok" or not second.answer:
            info["note"] = f"Escalation to {second.key} failed; kept first answer."
            trace["cascade"] = info
            return first
        second_check = self.checker.check(second.answer, question, images, classification, evidence)
        info["escalated_risk"] = second_check.risk
        info["escalated_check"] = second_check.to_dict()
        keep_second = second_check.risk <= first_check.risk
        info["kept"] = second.key if keep_second else first.key
        trace["cascade"] = info
        if keep_second:
            trace["answer_check"] = second_check.to_dict()
            return second
        return first

    def compare(self, question: str, images: Sequence[ImageSource], trace: dict[str, Any],
                run_kwargs: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Run cheapest eligible, strongest, and the adaptive choice with identical prompts."""
        n = len(images)
        adaptive = (trace.get("result") or {}).get("key") or trace["routing"]["selected_key"]
        strategies = [
            ("cheapest_eligible", self.policy.cheapest_eligible(n)),
            ("strongest", self.policy.strongest(n)),
            ("jev_adaptive", adaptive),
        ]
        cache: dict[str, dict[str, Any]] = {}
        for a in trace.get("attempts", []):
            if a.get("status") == "ok":
                cache.setdefault(a["key"], a)
        rows = []
        for name, key in strategies:
            if key is None:
                rows.append({"strategy": name, "key": None, "result": None})
                continue
            if key not in cache:
                cache[key] = self.executor.run(key, question, images, **(run_kwargs or {})).to_dict()
            rows.append({"strategy": name, "key": key, "result": cache[key]})
        return rows

    def save_trace(self, trace: dict[str, Any]) -> Optional[Path]:
        if self.trace_dir is None:
            return None
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{datetime.now(timezone.utc):%Y%m%d}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace, default=str) + "\n")
        return path
