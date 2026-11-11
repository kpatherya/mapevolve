"""Deterministic model selection.

Jev only produces a `QueryClassification`. Everything here is a pure function
of that classification, the registry, the image count, and the user's
preference, so a routing decision is reproducible from its trace.

Expected quality marginalizes over Jev's full distributions instead of using
only the top label:

    Q(m) = sum_t sum_d  P(type=t) * P(difficulty=d) * q(m, t, d)

where q is the shrunk task x difficulty score from the registry.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from jev_router.quality_model import QualityModel, model_attributes
from jev_router.registry import ROOT, ModelRegistry
from jev_router.schemas import MODES, CandidateScore, QueryClassification, RoutingDecision

# Probabilities below this are ignored when marginalizing (keeps traces readable).
_MIN_PROB = 0.02


def _log_minmax(values: dict[str, float]) -> dict[str, float]:
    logs = {k: math.log10(max(v, 1e-9)) for k, v in values.items()}
    lo, hi = min(logs.values()), max(logs.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in logs}
    return {k: (v - lo) / (hi - lo) for k, v in logs.items()}


class RoutingPolicy:
    def __init__(self, registry: ModelRegistry, config: dict[str, Any],
                 quality_model: Optional[QualityModel] = None, use_quality_model: Optional[bool] = None):
        self.registry = registry
        self.config = config
        self.modes = config["modes"]
        self.low_conf = config.get("jev", {}).get("low_confidence_threshold", 0.35)
        qcfg = config.get("quality_model", {})
        self.kappa = qcfg.get("uncertainty_kappa", 1.0)
        if quality_model is None and qcfg.get("path") and use_quality_model is not False:
            quality_model = QualityModel.load(ROOT / qcfg["path"])
        enough = quality_model is not None and quality_model.meta.get("n_cases", 0) >= qcfg.get("min_cases", 30)
        active = use_quality_model if use_quality_model is not None else (enough or qcfg.get("force", False))
        self.quality_model = quality_model if (active and quality_model is not None) else None
        self.model_attrs = model_attributes(registry.data["models"]) if self.quality_model else {}
        self.quality_model_status = (
            "inactive (no model file)" if quality_model is None else
            f"active (trained on {quality_model.meta.get('n_cases')} cases)" if self.quality_model else
            f"inactive ({quality_model.meta.get('n_cases')} cases < min_cases {qcfg.get('min_cases', 30)})")

    # ------------------------------------------------------------------ #
    def score_candidates(
        self,
        classification: QueryClassification,
        n_images: int,
        budget_usd: Optional[float],
        penalty: float,
        require_listed: bool = True,
        allowed_keys: Optional[Iterable[str]] = None,
        risk_kappa: float = 0.0,
    ) -> list[CandidateScore]:
        allowed = set(allowed_keys) if allowed_keys is not None else None
        type_probs = {t: p for t, p in classification.query_type.probabilities.items() if p >= _MIN_PROB}
        diff_probs = {d: p for d, p in classification.difficulty.probabilities.items() if p >= _MIN_PROB}
        t_norm, d_norm = sum(type_probs.values()), sum(diff_probs.values())

        out = []
        for key in self.registry.keys():
            model = self.registry.get(key)
            cand = CandidateScore(key=key, model_id=model["model_id"], eligible=True)
            if allowed is not None and key not in allowed:
                cand.eligible = False
                cand.exclusion_reasons.append("not in allowed model set")
            if require_listed and not self.registry.is_listed(key):
                cand.eligible = False
                cand.exclusion_reasons.append(f"availability: {model['availability']['status']}")
            if not self.registry.supports_images(key, n_images):
                cand.eligible = False
                cand.exclusion_reasons.append(f"cannot take {n_images} image(s)")

            q_sum = u_sum = 0.0
            sources = set()
            have_scores = True
            for t, pt in type_probs.items():
                for d, pd in diff_probs.items():
                    est = self.registry.score(key, t, d)
                    if est.quality is None:
                        have_scores = False
                        break
                    w = (pt / t_norm) * (pd / d_norm)
                    q_sum += w * est.quality
                    u_sum += w * (est.unsupported_rate or 0.0)
                    if pt / t_norm >= 0.2 and pd / d_norm >= 0.2:
                        sources.add(est.source)
            std = 0.0
            if self.quality_model is not None:
                covered = set(self.quality_model.meta.get("query_types", []))
                p_cov = sum(p for t, p in type_probs.items() if t in covered) / t_norm
                if p_cov > 0:
                    pred = self.quality_model.predict(classification, key, self.model_attrs.get(key))
                    reg_q = q_sum if have_scores else pred.quality
                    reg_u = u_sum if have_scores else pred.unsupported
                    q_sum = p_cov * pred.quality + (1 - p_cov) * reg_q
                    u_sum = p_cov * pred.unsupported + (1 - p_cov) * reg_u
                    std = p_cov * pred.quality_std
                    have_scores = True
                    sources.add(f"{pred.source}, weight {p_cov:.2f}")
            if have_scores:
                cand.expected_quality = round(q_sum, 4)
                cand.unsupported_rate = round(u_sum, 4)
                cand.quality_std = round(std, 4) if self.quality_model is not None else None
                cand.adjusted_quality = round(q_sum - penalty * u_sum - risk_kappa * std, 4)
                cand.score_sources = sorted(sources)
            else:
                cand.eligible = False
                cand.exclusion_reasons.append("no evaluation data")

            cost, cost_source = self.registry.estimate_cost(key, n_images)
            cand.est_cost_usd = round(cost, 6) if cost is not None else None
            cand.cost_source = cost_source
            cand.est_latency_seconds = self.registry.latency(key)
            if budget_usd is not None:
                if cost is None:
                    cand.eligible = False
                    cand.exclusion_reasons.append("cost unknown under a budget")
                elif cost > budget_usd:
                    cand.eligible = False
                    cand.exclusion_reasons.append(f"est. cost ${cost:.4f} > budget ${budget_usd:.4f}")
            out.append(cand)
        return out

    # ------------------------------------------------------------------ #
    def route(
        self,
        classification: QueryClassification,
        n_images: int,
        mode: str = "balanced",
        budget_usd: Optional[float] = None,
        require_listed: bool = True,
        allowed_keys: Optional[Iterable[str]] = None,
    ) -> RoutingDecision:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        cfg = self.modes[mode]
        if budget_usd is None:
            budget_usd = self.config.get("default_budget_usd_per_query")
        cands = self.score_candidates(
            classification, n_images, budget_usd, cfg.get("unsupported_penalty", 0.0), require_listed, allowed_keys,
            risk_kappa=self.kappa if mode == "accuracy" else 0.0,
        )
        eligible = [c for c in cands if c.eligible]
        rationale: list[str] = []
        warnings = self._warnings(classification)

        if not eligible:
            return RoutingDecision(
                mode, None, None, [], ["No eligible model: every candidate was filtered out."],
                cands, warnings + ["no eligible model"], budget_usd,
            )

        ranked = getattr(self, f"_rank_{mode}")(eligible, cfg, rationale)
        best = ranked[0]
        rationale.insert(
            0,
            f"Jev ({classification.source}) classified the query as "
            f"{classification.query_type.label} / {classification.difficulty.label}; "
            f"{len(eligible)} of {len(cands)} models are eligible.",
        )
        rationale.append(
            f"Selected {best.key} ({best.model_id}): expected quality {best.expected_quality:.3f}, "
            f"unsupported-claim risk {best.unsupported_rate:.3f}, est. cost "
            f"${best.est_cost_usd:.5f}, mean latency {best.est_latency_seconds}s."
        )
        rationale.append("Score basis: " + "; ".join(best.score_sources))
        n_fallback = self.config.get("generation", {}).get("max_failover_models", 2)
        return RoutingDecision(
            mode=mode,
            selected_key=best.key,
            selected_model_id=best.model_id,
            fallback_keys=[c.key for c in ranked[1 : 1 + n_fallback]],
            rationale=rationale,
            candidates=cands,
            warnings=warnings,
            budget_usd=budget_usd,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _rank_economy(eligible: list[CandidateScore], cfg: dict, rationale: list[str]) -> list[CandidateScore]:
        floor = cfg.get("min_quality", 0.7)
        passing = [c for c in eligible if c.adjusted_quality >= floor]
        if passing:
            rationale.append(f"{len(passing)} model(s) meet the quality floor {floor:.2f}; choosing the cheapest.")
        else:
            best_q = max(c.adjusted_quality for c in eligible)
            margin = cfg.get("relax_margin", 0.05)
            passing = [c for c in eligible if c.adjusted_quality >= best_q - margin]
            rationale.append(
                f"No model meets the floor {floor:.2f} (best {best_q:.3f}); relaxed to models within "
                f"{margin:.2f} of the best and chose the cheapest."
            )
        rest = [c for c in eligible if c not in passing]
        key = lambda c: (c.est_cost_usd, -c.adjusted_quality)
        return sorted(passing, key=key) + sorted(rest, key=lambda c: -c.adjusted_quality)

    @staticmethod
    def _rank_balanced(eligible: list[CandidateScore], cfg: dict, rationale: list[str]) -> list[CandidateScore]:
        w = cfg["weights"]
        cost_n = _log_minmax({c.key: c.est_cost_usd for c in eligible})
        lat_n = _log_minmax({c.key: c.est_latency_seconds or 60.0 for c in eligible})
        for c in eligible:
            c.utility = round(
                w["quality"] * c.adjusted_quality - w["cost"] * cost_n[c.key] - w["latency"] * lat_n[c.key], 4
            )
        rationale.append(
            f"Utility = {w['quality']}*quality - {w['cost']}*cost - {w['latency']}*latency "
            "(cost and latency log-scaled to [0,1] across eligible models)."
        )
        return sorted(eligible, key=lambda c: (-c.utility, c.est_cost_usd))

    @staticmethod
    def _rank_accuracy(eligible: list[CandidateScore], cfg: dict, rationale: list[str]) -> list[CandidateScore]:
        tie = cfg.get("tie_margin", 0.01)
        best_q = max(c.adjusted_quality for c in eligible)
        rationale.append(
            f"Highest expected quality within budget; models within {tie:.2f} of the best are "
            "tie-broken by unsupported-claim risk, then cost."
        )
        top = [c for c in eligible if c.adjusted_quality >= best_q - tie]
        rest = [c for c in eligible if c not in top]
        return sorted(top, key=lambda c: (c.unsupported_rate, c.est_cost_usd)) + sorted(
            rest, key=lambda c: -c.adjusted_quality
        )

    def _warnings(self, c: QueryClassification) -> list[str]:
        out = []
        if c.source != "jev":
            out.append(f"Jev was not used ({c.source}): {c.error or 'n/a'}")
        if c.difficulty.confidence is not None and c.difficulty.confidence < self.low_conf:
            out.append(f"Low Jev confidence on difficulty ({c.difficulty.confidence:.2f}).")
        flags = c.flags()
        if flags["needs_image_retrieval"]:
            out.append("Query likely needs image retrieval, which is not implemented in this prototype.")
        if flags["needs_pair_verification"]:
            out.append("Query likely needs pair verification, which is not implemented; the VLM is told to be cautious.")
        if flags["needs_external_tools"]:
            out.append("Query likely needs external records or tools; answers are limited to the supplied images.")
        return out

    # ------------------------------------------------------------------ #
    # Baselines used by the UI comparison and the offline evaluation.
    # ------------------------------------------------------------------ #
    def cheapest_eligible(self, n_images: int, require_listed: bool = True, allowed_keys=None) -> Optional[str]:
        cands = [k for k in self._capable(n_images, require_listed, allowed_keys)]
        costs = {k: self.registry.estimate_cost(k, n_images)[0] for k in cands}
        costs = {k: v for k, v in costs.items() if v is not None}
        return min(costs, key=costs.get) if costs else None

    def strongest(self, n_images: int, require_listed: bool = True, allowed_keys=None) -> Optional[str]:
        """One universal ranking by general score (ties: lower unsupported rate)."""
        cands = [k for k in self._capable(n_images, require_listed, allowed_keys) if self.registry.general_quality(k) is not None]
        if not cands:
            return None
        def key(k):
            ev = self.registry.get(k)["evaluation"]["general"]
            return (-ev["normalized"], ev["unsupported_claim_rate"])
        return sorted(cands, key=key)[0]

    def fixed_task(self, task: str, n_images: int, require_listed: bool = True, allowed_keys=None) -> Optional[str]:
        """Static task -> model table: best task-level score, ignoring difficulty."""
        cands = self._capable(n_images, require_listed, allowed_keys)
        scored = []
        for k in cands:
            ev = self.registry.get(k).get("evaluation")
            if not ev:
                continue
            ts = ev["task_scores"].get(task, {}).get("overall") or ev["general"]
            scored.append((-ts["normalized"], ts["unsupported_claim_rate"], k))
        return sorted(scored)[0][2] if scored else None

    def _capable(self, n_images: int, require_listed: bool, allowed_keys) -> list[str]:
        allowed = set(allowed_keys) if allowed_keys is not None else None
        return [
            k
            for k in self.registry.keys()
            if (allowed is None or k in allowed)
            and (not require_listed or self.registry.is_listed(k))
            and self.registry.supports_images(k, n_images)
        ]
