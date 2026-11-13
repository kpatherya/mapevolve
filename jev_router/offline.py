"""Replay routing decisions against recorded, graded answers.

Shared by the Pareto report (router curve) and the tuning harness. For each
case the registry is rebuilt from every *other* case (leave-one-case-out), so
a case's own outcome never informs its route. Only cases graded for every
pool model are used, so every strategy is scored on the same cases.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Optional

from jev_router.benchmark_data import model_metrics
from jev_router.classify_cache import classify_cases, resolve_image_aware
from jev_router.dataset import Case, load_all
from jev_router.graded import Graded, as_records, load_graded, row_cost
from jev_router.policy import RoutingPolicy
from jev_router.registry import ModelRegistry, load_routing_config
from jev_router.schemas import QueryClassification


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


@dataclass
class OfflineData:
    cfg: dict[str, Any]
    manifest: dict[str, Case]
    by_case: dict[str, dict[str, Graded]]
    pool: list[str]
    case_ids: list[str]
    classifications: dict[str, QueryClassification]
    base_registry: ModelRegistry
    records: list[Any]
    _registries: dict[str, ModelRegistry] = field(default_factory=dict)

    def registry_for(self, cid: str, loo: bool = True) -> ModelRegistry:
        key = cid if loo else "__all__"
        if key not in self._registries:
            recs = [r for r in self.records if not (loo and r.case_id == cid)]
            self._registries[key] = self.base_registry.with_metrics(model_metrics(recs))
        return self._registries[key]


def load_offline(classifier: str = "heuristic", image_aware: Optional[bool] = None, split: Optional[str] = None,
                 cfg: Optional[dict[str, Any]] = None, registry_split: Optional[str] = None) -> OfflineData:
    """`split` selects the cases to route; `registry_split` restricts which graded
    cases may inform the registry (use "dev" when tuning or testing, so test
    grades never shape the scores being tuned)."""
    cfg = cfg or load_routing_config()
    manifest = {c.case_id: c for c in load_all()}
    graded = [g for g in load_graded() if g.case_id in manifest]
    by_case: dict[str, dict[str, Graded]] = defaultdict(dict)
    for g in graded:
        by_case[g.case_id][g.model_key] = g
    pool = sorted({g.model_key for g in graded})
    case_ids = sorted(c for c, ms in by_case.items() if set(ms) >= set(pool)
                      and (split is None or manifest[c].split == split))
    image_aware = resolve_image_aware(image_aware, classifier)
    cls = classify_cases([manifest[c] for c in case_ids], classifier, image_aware) if case_ids else {}
    reg_rows = [g for g in graded if registry_split is None or manifest[g.case_id].split == registry_split]
    return OfflineData(cfg, manifest, by_case, pool, case_ids, cls, ModelRegistry.load(routing=cfg),
                       as_records(reg_rows, manifest))


def route_all(data: OfflineData, cfg: dict[str, Any], mode: str, budget: Optional[float] = None,
              loo: bool = True, require_listed: bool = False) -> dict[str, Optional[str]]:
    k = cfg.get("registry", {}).get("shrinkage_k", 2)
    out = {}
    for cid in data.case_ids:
        reg = data.registry_for(cid, loo)
        reg.shrinkage_k = k
        policy = RoutingPolicy(reg, cfg, use_quality_model=False)
        n_images = len(data.manifest[cid].images)
        out[cid] = policy.route(data.classifications[cid], n_images, mode, budget, require_listed, data.pool).selected_key
    return out


def fixed_choice(data: OfflineData, key: str) -> dict[str, str]:
    return {cid: key for cid in data.case_ids}


def score_choices(data: OfflineData, choices: dict[str, Optional[str]]) -> dict[str, Any]:
    rows = []
    for cid, key in choices.items():
        g = data.by_case[cid].get(key) if key else None
        if g is None:
            continue
        cost, _ = row_cost(g, data.base_registry)
        best = max(x.score for x in data.by_case[cid].values())
        rows.append((g, cost, best))
    if not rows:
        return {"n_cases": 0}
    costs = [c for _, c, _ in rows if c is not None]
    return {
        "n_cases": len(rows),
        "mean_score_0_4": round(mean(g.score for g, _, _ in rows), 4),
        "quality": round(mean(g.score / 4 for g, _, _ in rows), 4),
        "correct_rate": round(mean(g.score >= 4 for g, _, _ in rows), 4),
        "unsupported_rate": round(mean(g.unsupported for g, _, _ in rows), 4),
        "avg_cost_usd": round(mean(costs), 7) if len(costs) == len(rows) else None,
        "avg_latency_s": round(mean(g.latency_seconds or 0 for g, _, _ in rows), 3),
        "mean_regret": round(mean(b - g.score for g, _, b in rows), 4),
        "models_used": "|".join(sorted({g.model_key for g, _, _ in rows})),
    }
