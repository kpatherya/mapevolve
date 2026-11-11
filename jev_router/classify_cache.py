"""Classify manifest cases once per configuration and cache the result.

Shared by the classifier evaluation, predictor training and offline routing
evaluation so Jev is never called twice for the same inputs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from jev_router.classifier import JevQueryClassifier
from jev_router.dataset import ROOT, Case, routing_inputs
from jev_router.registry import load_routing_config
from jev_router.schemas import QueryClassification

CACHE = ROOT / "results" / "classifications"
CACHE_VERSION = "v2"  # bump when classifier inputs or rules change


def input_hash(case: Case, tag: str) -> str:
    h = hashlib.sha256(f"{CACHE_VERSION}:{tag}".encode())
    h.update(case.question.encode())
    h.update((case.supplied_evidence or "").encode())
    for p in case.image_paths():
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


def classify_cases(cases: list[Case], mode: str, image_aware: bool = False,
                   refresh: bool = False) -> dict[str, QueryClassification]:
    """mode: "jev" (falls back per case on error) or "heuristic"."""
    tag = f"{mode}{'_img' if image_aware else ''}"
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{tag}.jsonl"
    cache: dict[str, dict[str, Any]] = {}
    if path.exists() and not refresh:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[row["input_hash"]] = row
    cfg = load_routing_config().get("jev", {})
    clf = JevQueryClassifier(model=cfg.get("model", "jev-latest"), timeout=cfg.get("timeout_seconds", 20),
                             use_jev=(mode == "jev"))
    out, new = {}, []
    expected_source = "jev" if mode == "jev" else "heuristic_fallback"
    for case in cases:
        key = input_hash(case, tag)
        if key in cache:
            out[case.case_id] = QueryClassification.from_dict(cache[key]["classification"])
            continue
        inp = routing_inputs(case)
        cls = clf.classify(inp["question"], inp["images"], image_aware=image_aware, evidence=inp["supplied_evidence"])
        out[case.case_id] = cls
        if cls.source == expected_source:  # never cache a Jev failure as a Jev answer
            new.append({"case_id": case.case_id, "input_hash": key, "classification": cls.to_dict()})
    if refresh and path.exists():
        path.unlink()
    with path.open("a", encoding="utf-8") as fh:
        for row in new:
            fh.write(json.dumps(row, default=str) + "\n")
    return out
