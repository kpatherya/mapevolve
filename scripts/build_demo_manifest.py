"""Build datasets/demo_world.json from data/demo_world/cases.yaml.

    python -m scripts.build_demo_manifest
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo_world"
OUT = ROOT / "datasets" / "demo_world.json"

# Whole locations go to test so no place appears in both splits.
TEST_LOCATIONS = {"frauenkirche", "westminster", "istanbul", "sydney", "flatiron", "sagrada-familia", "casa-batllo"}


def _labels(c: dict) -> dict:
    """Review status. A reviewer sets `reviewed: true` and `reviewer: <name>` on the
    case in cases.yaml (and may add `review_note`); the draft author stays recorded."""
    reviewed = bool(c.get("reviewed", False))
    labels = {"author": "draft_by_claude", "reviewed": reviewed}
    if reviewed:
        if not c.get("reviewer"):
            raise SystemExit(f"{c['id']}: reviewed: true needs a reviewer name")
        labels["reviewer"] = c["reviewer"]
    else:
        labels["note"] = "Drafted after viewing the images; needs human review."
    if c.get("review_note"):
        labels["review_note"] = c["review_note"]
    return labels


def main() -> int:
    spec = yaml.safe_load((DEMO / "cases.yaml").read_text(encoding="utf-8"))
    defaults = spec["defaults"]
    attribution = {r["id"]: r for r in json.loads((DEMO / "attribution.json").read_text(encoding="utf-8"))}
    cases = []
    for c in spec["cases"]:
        answerable = c.get("answerable", True)
        action = c.get("action") or ("answer" if answerable else "decline")
        images = []
        for image_id in c["images"]:
            if image_id not in attribution:
                raise SystemExit(f"{c['id']}: image {image_id} has no attribution record")
            images.append(f"data/demo_world/images/{image_id}.jpg")
        cases.append({
            "case_id": f"W-{c['id']}",
            "dataset": "demo_world",
            "location_id": c["location"],
            "split": "test" if c["location"] in TEST_LOCATIONS else "dev",
            "query_type": c["type"],
            "difficulty": c["difficulty"],
            "answerable": answerable,
            "expected_action": action,
            "question": defaults.get(c["question"], c["question"]),
            "images": images,
            "reference": {
                "answer": " ".join(c["answer"].split()),
                "required_points": c.get("required_points", []),
                "acceptable_confounders": c.get("confounders", []),
                "unsupported_claims_to_avoid": c.get("avoid", []),
            },
            "labels": _labels(c),
            "hard_negative": bool(c.get("hard_negative", False)),
            "supplied_evidence": c.get("evidence"),
            "tags": c.get("tags", []),
        })
    OUT.write_text(json.dumps({"schema_version": "2.0", "dataset": "demo_world",
                               "attribution": "data/demo_world/attribution.json", "cases": cases},
                              indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
