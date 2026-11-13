"""Pareto frontiers of VLMs per query cell, with bootstrap membership.

A model is *dominated* in a cell when another model is at least as good on
every objective and strictly better on one. The non-dominated models form the
frontier: the only sensible choices for some trade-off between the objectives.

Cells: "all", each query type ("type=<t>"), and each type x difficulty
("type=<t>|difficulty=<d>"), using manifest labels (this is offline
characterisation of models, like the registry; routing still uses Jev's
predicted cell).

Objectives (direction): quality (max), unsupported_rate (min), cost (min),
latency (min). Optional axes are reported when data exists but join the
frontier only when present for every model in the cell: unanswerable_quality
(max; refusal calibration), reliability (max; ok rate in live runs),
inconsistency (min; score spread across repeated runs), latency_p90 (min).

With few cases a single frontier is fragile, so `membership` = share of
case-level bootstrap resamples in which the model is on the frontier.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Optional

import numpy as np

from jev_router.dataset import ROOT

CORE_OBJECTIVES: list[tuple[str, str]] = [
    ("quality", "max"), ("unsupported_rate", "min"), ("cost_usd", "min"), ("latency_s", "min"),
]
OPTIONAL_OBJECTIVES: list[tuple[str, str]] = [
    ("unanswerable_quality", "max"), ("reliability", "max"), ("inconsistency", "min"), ("latency_p90_s", "min"),
]
LIVE_DIR = ROOT / "results" / "live_runs"


@dataclass
class Obs:
    """One graded answer with what the frontier needs."""
    case_id: str
    model: str
    score: float           # 0..4
    unsupported: bool
    cost: Optional[float]
    latency: Optional[float]
    answerable: bool
    run_id: Optional[str] = None


@dataclass
class ModelPoint:
    model: str
    n_cases: int
    metrics: dict[str, Optional[float]]
    on_frontier: bool = False
    membership: Optional[float] = None
    dominated_by: list[str] = field(default_factory=list)


def live_reliability(live_dir: Path = LIVE_DIR) -> dict[str, dict[str, float]]:
    """Per-model ok rate and JSON-format compliance from live run logs."""
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0, "json": 0})
    if live_dir.exists():
        for path in live_dir.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                s = stats[r["model_key"]]
                s["n"] += 1
                s["ok"] += r.get("status") == "ok"
                s["json"] += bool(r.get("structured"))
    return {m: {"reliability": s["ok"] / s["n"], "json_compliance": s["json"] / s["n"], "n_calls": s["n"]}
            for m, s in stats.items() if s["n"]}


def summarize(obs: list[Obs], reliability: Optional[dict[str, dict[str, float]]] = None) -> dict[str, ModelPoint]:
    by_model: dict[str, list[Obs]] = defaultdict(list)
    for o in obs:
        by_model[o.model].append(o)
    out = {}
    for m, rows in by_model.items():
        costs = [o.cost for o in rows if o.cost is not None]
        lats = [o.latency for o in rows if o.latency is not None]
        unans = [o.score / 4 for o in rows if not o.answerable]
        per_case: dict[str, list[float]] = defaultdict(list)
        for o in rows:
            per_case[o.case_id].append(o.score / 4)
        spreads = [pstdev(v) for v in per_case.values() if len(v) > 1]
        metrics = {
            "quality": mean(o.score / 4 for o in rows),
            "correct_rate": mean(o.score >= 4 for o in rows),
            "unsupported_rate": mean(o.unsupported for o in rows),
            "cost_usd": mean(costs) if len(costs) == len(rows) else None,
            "latency_s": mean(lats) if lats else None,
            "latency_p90_s": float(np.percentile(lats, 90)) if len(lats) >= 10 else None,
            "unanswerable_quality": mean(unans) if unans else None,
            "inconsistency": mean(spreads) if spreads else None,
            "reliability": (reliability or {}).get(m, {}).get("reliability"),
        }
        out[m] = ModelPoint(m, len(per_case), {k: (round(v, 6) if v is not None else None) for k, v in metrics.items()})
    return out


def active_objectives(points: dict[str, ModelPoint], extra: bool = True) -> list[tuple[str, str]]:
    objs = [o for o in CORE_OBJECTIVES if all(p.metrics.get(o[0]) is not None for p in points.values())]
    if extra:
        objs += [o for o in OPTIONAL_OBJECTIVES if all(p.metrics.get(o[0]) is not None for p in points.values())]
    return objs


def dominates(a: dict[str, Any], b: dict[str, Any], objectives: list[tuple[str, str]]) -> bool:
    better = False
    for name, direction in objectives:
        x, y = a[name], b[name]
        if direction == "max":
            x, y = -x, -y
        if x > y + 1e-12:
            return False
        if x < y - 1e-12:
            better = True
    return better


def frontier(points: dict[str, ModelPoint], objectives: list[tuple[str, str]]) -> set[str]:
    on = set()
    for m, p in points.items():
        doms = [o for o, q in points.items() if o != m and dominates(q.metrics, p.metrics, objectives)]
        p.dominated_by = sorted(doms)
        if not doms:
            on.add(m)
    return on


def analyze_cell(obs: list[Obs], objectives: Optional[list[tuple[str, str]]] = None, n_boot: int = 200,
                 seed: int = 0, reliability: Optional[dict] = None) -> dict[str, Any]:
    points = summarize(obs, reliability)
    objs = objectives or active_objectives(points)
    on = frontier(points, objs)
    for m, p in points.items():
        p.on_frontier = m in on
    cases = sorted({o.case_id for o in obs})
    by_case: dict[str, list[Obs]] = defaultdict(list)
    for o in obs:
        by_case[o.case_id].append(o)
    rng = np.random.default_rng(seed)
    counts = {m: 0 for m in points}
    valid = 0
    for _ in range(n_boot):
        sample = rng.choice(cases, size=len(cases), replace=True)
        boot = [Obs(f"{o.case_id}#{i}", o.model, o.score, o.unsupported, o.cost, o.latency, o.answerable, o.run_id)
                for i, c in enumerate(sample) for o in by_case[c]]
        bp = summarize(boot, reliability)
        if set(bp) != set(points):
            continue
        valid += 1
        for m in frontier(bp, objs):
            counts[m] += 1
    for m, p in points.items():
        p.membership = round(counts[m] / valid, 3) if valid else None
    return {"n_cases": len(cases), "objectives": objs, "frontier": sorted(on),
            "points": {m: asdict(p) for m, p in sorted(points.items())}}


def cells_for(obs: list[Obs], case_meta: dict[str, Any]) -> dict[str, list[Obs]]:
    cells: dict[str, list[Obs]] = defaultdict(list)
    for o in obs:
        c = case_meta[o.case_id]
        cells["all"].append(o)
        cells[f"type={c.query_type}"].append(o)
        cells[f"type={c.query_type}|difficulty={c.difficulty}"].append(o)
    return cells


def complete_cases(obs: list[Obs], models: Optional[Iterable[str]] = None) -> list[Obs]:
    """Keep only cases graded for every model, so models are compared on the same cases."""
    pool = set(models) if models else {o.model for o in obs}
    have: dict[str, set[str]] = defaultdict(set)
    for o in obs:
        have[o.case_id].add(o.model)
    keep = {c for c, ms in have.items() if pool <= ms}
    return [o for o in obs if o.case_id in keep and o.model in pool]


def load_frontier_table(path: Path) -> Optional[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cell_lookup(table: dict[str, Any], query_type: str, difficulty: str, min_cases: int) -> Optional[tuple[str, dict]]:
    """Most specific cell with enough cases: type x difficulty, then type, then all."""
    for key in (f"type={query_type}|difficulty={difficulty}", f"type={query_type}", "all"):
        cell = table.get("cells", {}).get(key)
        if cell and cell["n_cases"] >= min_cases:
            return key, cell
    return None


def quality_cost_front(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """2-D frontier (max quality, min cost) for plotting, sorted by cost."""
    pts = sorted(points, key=lambda p: (p[2], -p[1]))
    front, best = [], -math.inf
    for name, q, c in pts:
        if q > best + 1e-12:
            front.append((name, q, c))
            best = q
    return front
