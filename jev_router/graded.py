"""Graded VLM responses in one format, keyed by manifest case IDs.

Sources:
  * the Georgia Tech pilot (benchmark/artifacts/judged_responses.csv, with
    manual_review.csv human overrides) -> case IDs "GT-<id>"
  * results/graded/*.jsonl written by scripts.run_live_benchmark +
    scripts.grade_responses (judge or human scores)

Each row: case_id, model_key, score (0-4), label, unsupported, score_source,
latency_seconds, cost_usd.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from jev_router.benchmark_data import load_responses
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


def load_graded(include_live: bool = True) -> list[Graded]:
    rows = [
        Graded(f"GT-{r.case_id}", r.model_key, r.score, r.label, r.unsupported_claim, r.score_source,
               r.latency_seconds, r.reported_cost_usd, r.input_tokens, r.output_tokens, r.response_text)
        for r in load_responses()
    ]
    if include_live and GRADED_DIR.exists():
        latest: dict[tuple[str, str], Graded] = {}
        for path in sorted(GRADED_DIR.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                g = Graded(**{k: d.get(k) for k in Graded.__dataclass_fields__})
                # Human scores beat judge scores for the same (case, model).
                prev = latest.get((g.case_id, g.model_key))
                if prev is None or g.score_source == "human" or prev.score_source != "human":
                    latest[(g.case_id, g.model_key)] = g
        rows += list(latest.values())
    return rows


def append_graded(path: Path, rows: list[Graded]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")
