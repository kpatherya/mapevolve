"""Benchmark case manifest (schema v2): loading, validation, coverage.

One manifest format covers every query type and dataset (Georgia Tech pilot,
world-landmark demo, future sets). Only `question`, `images` and
`supplied_evidence` are user inputs; every other field is label data that
routing code must never read (see `routing_inputs`).

Splits are assigned per `location_id` so the same place (and therefore the
same or near-duplicate images) never appears in both dev and test.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from jev_router.schemas import DIFFICULTIES, QUERY_TYPES

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

SPLITS = ("dev", "test")
ACTIONS = ("answer", "verify_then_answer", "retrieve_then_answer", "use_tools", "decline")
LABEL_AUTHORS = ("human", "human_ground_truth_file", "draft_by_claude")


@dataclass
class Case:
    case_id: str
    dataset: str
    location_id: str
    split: str
    query_type: str
    difficulty: str
    answerable: bool                    # closed-world: from the supplied images/evidence alone
    expected_action: str
    question: str
    images: list[str]                   # repo-relative paths
    reference: dict[str, Any]
    labels: dict[str, Any]
    hard_negative: bool = False
    supplied_evidence: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Case":
        return cls(**d)

    def image_paths(self) -> list[Path]:
        return [ROOT / p for p in self.images]


def routing_inputs(case: Case) -> dict[str, Any]:
    """The only parts of a case that classifiers, policies and VLMs may see."""
    return {
        "question": case.question,
        "images": case.image_paths(),
        "supplied_evidence": case.supplied_evidence,
    }


def load_manifest(path: Path) -> list[Case]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Case.from_dict(c) for c in data["cases"]]


def load_all(paths: Optional[Iterable[Path]] = None) -> list[Case]:
    paths = list(paths) if paths else sorted(DATASETS.glob("*.json"))
    cases: list[Case] = []
    for p in paths:
        cases.extend(load_manifest(p))
    return cases


def validate(cases: list[Case], check_files: bool = True) -> list[str]:
    """Return a list of problems (empty means valid)."""
    errors: list[str] = []
    ids = Counter(c.case_id for c in cases)
    errors += [f"duplicate case_id {k}" for k, n in ids.items() if n > 1]
    loc_split: dict[str, set[str]] = defaultdict(set)
    img_split: dict[str, set[str]] = defaultdict(set)
    for c in cases:
        where = f"{c.case_id}:"
        if c.split not in SPLITS:
            errors.append(f"{where} split {c.split!r} not in {SPLITS}")
        if c.query_type not in QUERY_TYPES:
            errors.append(f"{where} query_type {c.query_type!r} not in {QUERY_TYPES}")
        if c.difficulty not in DIFFICULTIES:
            errors.append(f"{where} difficulty {c.difficulty!r} not in {DIFFICULTIES}")
        if c.expected_action not in ACTIONS:
            errors.append(f"{where} expected_action {c.expected_action!r} not in {ACTIONS}")
        if not c.answerable and c.expected_action in ("answer", "verify_then_answer"):
            errors.append(f"{where} unanswerable case cannot expect action {c.expected_action!r}")
        if not 1 <= len(c.images) <= 4:
            errors.append(f"{where} needs 1-4 images, has {len(c.images)}")
        if not c.question.strip():
            errors.append(f"{where} empty question")
        if not c.reference.get("answer"):
            errors.append(f"{where} reference.answer is required")
        if c.labels.get("author") not in LABEL_AUTHORS:
            errors.append(f"{where} labels.author must be one of {LABEL_AUTHORS}")
        if check_files:
            for p in c.image_paths():
                if not p.is_file():
                    errors.append(f"{where} missing image {p.relative_to(ROOT)}")
        loc_split[f"{c.dataset}/{c.location_id}"].add(c.split)
        for img in c.images:
            img_split[img].add(c.split)
    errors += [f"location {k} appears in splits {sorted(v)}" for k, v in loc_split.items() if len(v) > 1]
    errors += [f"image {k} appears in splits {sorted(v)}" for k, v in img_split.items() if len(v) > 1]
    return errors


def coverage(cases: list[Case], min_per_cell: int = 5) -> dict[str, Any]:
    """Counts per query_type x difficulty x split, with thin cells flagged."""
    grid: dict[str, dict[str, dict[str, int]]] = {
        t: {d: {s: 0 for s in SPLITS} for d in DIFFICULTIES} for t in QUERY_TYPES
    }
    for c in cases:
        if c.query_type in grid and c.difficulty in DIFFICULTIES and c.split in SPLITS:
            grid[c.query_type][c.difficulty][c.split] += 1
    thin = [
        {"query_type": t, "difficulty": d, "split": s, "n": n, "needed": min_per_cell - n}
        for t, row in grid.items()
        for d, splits in row.items()
        for s, n in splits.items()
        if n < min_per_cell
    ]
    return {
        "n_cases": len(cases),
        "by_dataset": dict(Counter(c.dataset for c in cases)),
        "by_split": dict(Counter(c.split for c in cases)),
        "answerable": dict(Counter("answerable" if c.answerable else "unanswerable" for c in cases)),
        "expected_action": dict(Counter(c.expected_action for c in cases)),
        "hard_negatives": sum(c.hard_negative for c in cases),
        "label_review": dict(Counter(("reviewed" if c.labels.get("reviewed") else "unreviewed") + f":{c.labels.get('author')}" for c in cases)),
        "grid": grid,
        "min_per_cell": min_per_cell,
        "thin_cells": thin,
    }


def coverage_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark coverage ({report['n_cases']} cases)",
        "",
        f"Datasets: {report['by_dataset']} · splits: {report['by_split']} · "
        f"answerability: {report['answerable']} · hard negatives: {report['hard_negatives']}",
        "",
        f"Label review: {report['label_review']}",
        "",
        f"Cells show dev/test counts; ⚠ marks cells below {report['min_per_cell']} in either split.",
        "",
        "| Query type | " + " | ".join(DIFFICULTIES) + " |",
        "| --- | " + " | ".join("---" for _ in DIFFICULTIES) + " |",
    ]
    m = report["min_per_cell"]
    for t, row in report["grid"].items():
        cells = []
        for d in DIFFICULTIES:
            dev, test = row[d]["dev"], row[d]["test"]
            cells.append(f"{dev}/{test}" + (" ⚠" if min(dev, test) < m else ""))
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", f"Expected actions: {report['expected_action']}", ""]
    return "\n".join(lines)
