"""Graded VLM responses in one format, keyed by manifest case IDs.

Sources:
  * the Georgia Tech pilot (benchmark/artifacts/judged_responses.csv, with
    manual_review.csv human overrides) -> case IDs "GT-<id>"
  * results/graded/*.jsonl written by scripts.run_live_benchmark +
    scripts.grade_responses (judge or human scores)

Each row: case_id, model_key, score (0-4), label, unsupported, score_source,
latency_seconds, cost_usd, run_id. Several live runs of the same (case, model)
are repetitions; `load_graded(keep_repeats=True)` keeps them (for consistency
and latency spread), the default collapses to one row per (case, model).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from jev_router.benchmark_data import ResponseRecord, load_responses
from jev_router.dataset import ROOT

GRADED_DIR = ROOT / "results" / "graded"


@dataclass
class Graded:
    case_id: str
    model_key: str
    score: float
    label: str
    unsupported: bool
    score_source: str
    latency_seconds: Optional[float] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    response_text: str = ""
    run_id: Optional[str] = None


def load_graded(include_live: bool = True, keep_repeats: bool = False) -> list[Graded]:
    rows = [
        Graded(f"GT-{r.case_id}", r.model_key, r.score, r.label, r.unsupported_claim, r.score_source,
               r.latency_seconds, r.reported_cost_usd, r.input_tokens, r.output_tokens, r.response_text,
               run_id="pilot-20261013")
        for r in load_responses()
    ]
    if include_live and GRADED_DIR.exists():
        latest: dict[tuple, Graded] = {}
        for path in sorted(GRADED_DIR.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                g = Graded(**{k: d.get(k) for k in Graded.__dataclass_fields__ if k in d})
                key = (g.case_id, g.model_key, g.run_id) if keep_repeats else (g.case_id, g.model_key)
                # Human scores beat judge scores for the same key.
                prev = latest.get(key)
                if prev is None or g.score_source == "human" or prev.score_source != "human":
                    latest[key] = g
        rows += list(latest.values())
    return rows


def row_cost(g: Graded, registry: Any) -> tuple[Optional[float], str]:
    """Provider-reported cost, else recorded tokens x current registry price."""
    if g.cost_usd is not None:
        return g.cost_usd, "recorded_provider_cost"
    pricing = registry.get(g.model_key).get("pricing") if g.model_key in registry.models else None
    if pricing and pricing.get("input_usd_per_mtok") is not None and g.input_tokens is not None:
        return ((g.input_tokens * pricing["input_usd_per_mtok"]
                 + (g.output_tokens or 0) * (pricing.get("output_usd_per_mtok") or 0)) / 1e6,
                "estimated: recorded tokens x current reference price")
    return None, "unavailable"


def as_records(rows: list[Graded], cases: dict[str, Any]) -> list[ResponseRecord]:
    """Adapt graded rows (any source) to the registry's metric input, using
    manifest labels for task type and difficulty. Rows for unknown cases are dropped."""
    out = []
    for g in rows:
        c = cases.get(g.case_id)
        if c is None:
            continue
        out.append(ResponseRecord(
            case_id=g.case_id, difficulty=c.difficulty, task_type=c.query_type, model_key=g.model_key,
            model_id=g.model_key, score=g.score, label=g.label, unsupported_claim=g.unsupported,
            score_source=g.score_source, latency_seconds=g.latency_seconds, reported_cost_usd=g.cost_usd,
            input_tokens=g.input_tokens, output_tokens=g.output_tokens, response_text=g.response_text))
    return out


def append_graded(path: Path, rows: list[Graded]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")
