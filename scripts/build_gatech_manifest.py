"""Convert every existing Georgia Tech case into the v2 manifest.

Sources (references are copied, never rewritten):
  * benchmark/cases.json            6 graded temporal pairs (human ground-truth files)
  * benchmark/smoke_cases.json      4 smoke cases written by the team
  * benchmark/runs/building_query_results.json  2 adversarial cross-view cases

    python -m scripts.build_gatech_manifest
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "gatech_pilot.json"

# Split by location so no building appears in both splits. Chosen by hand to
# keep both splits non-empty; with this little data it is not balanced.
TEST_LOCATIONS = {"kendeda", "kessler-campanile", "carnegie", "brittain-dining-hall"}

PILOT_LOCATIONS = {"E1": "tech-tower", "E2": "kendeda", "M1": "klaus", "M2": "kessler-campanile",
                   "H1": "tech-tower", "H2": "carnegie"}


def split_for(location: str) -> str:
    return "test" if location in TEST_LOCATIONS else "dev"


def ground_truth_answer(raw: str) -> str:
    match = re.search(r"Ground truth:\s*(.+?)(?:\n|$)", raw)
    return match.group(1).strip() if match else raw.strip()


def pilot_cases() -> list[dict]:
    out = []
    for c in json.loads((ROOT / "benchmark" / "cases.json").read_text(encoding="utf-8")):
        loc = PILOT_LOCATIONS[c["case_id"]]
        rubric = c["rubric"]
        out.append({
            "case_id": f"GT-{c['case_id']}",
            "dataset": "gatech_pilot",
            "location_id": loc,
            "split": split_for(loc),
            "query_type": "temporal_comparison",
            "difficulty": {"medium": "moderate"}.get(c["difficulty"], c["difficulty"]),
            "answerable": True,
            "expected_action": "answer",
            "question": c["question"],
            "images": [f"benchmark/{c['image_a']}", f"benchmark/{c['image_b']}"],
            "reference": {
                "answer": ground_truth_answer(c["ground_truth_raw"]),
                "required_points": rubric.get("core_expectations", []),
                "acceptable_confounders": rubric.get("transient_or_confounder_expectations", []),
                "unsupported_claims_to_avoid": rubric.get("unsupported_claims_to_avoid", []),
            },
            "labels": {"author": "human_ground_truth_file", "reviewed": True,
                       "source": c["source_text"], "graded_responses": "benchmark/artifacts/judged_responses.csv"},
            "hard_negative": False,
            "supplied_evidence": None,
            "tags": ["graded_pilot"],
        })
    return out


SMOKE_MAP = {
    "simple-brittain-same-view": ("cross_view_matching", "easy", "answer", "brittain-dining-hall"),
    "moderate-tech-tower-different-view": ("cross_view_matching", "moderate", "verify_then_answer", "tech-tower"),
    "hard-klaus-record-fact": ("factual_retrieval", "hard", "use_tools", "klaus"),
    "hardest-two-buildings-near-klaus": ("spatial_reasoning", "hard", "use_tools", "tech-tower"),
}


def smoke_cases() -> list[dict]:
    out = []
    for c in json.loads((ROOT / "benchmark" / "smoke_cases.json").read_text(encoding="utf-8")):
        qtype, diff, action, loc = SMOKE_MAP[c["case_id"]]
        ref = c["reference"]
        out.append({
            "case_id": f"GT-{c['case_id']}",
            "dataset": "gatech_pilot",
            "location_id": loc,
            "split": split_for(loc),
            "query_type": qtype,
            "difficulty": diff,
            "answerable": bool(c["answerable"]),
            "expected_action": action,
            "question": c["query"],
            "images": c["images"],
            "reference": {
                "answer": f"{ref['expected_decision']}. Building(s): {ref['building']}.",
                "required_points": [ref["required_evidence"]],
                "unsupported_claims_to_avoid": [],
            },
            "labels": {"author": "human", "reviewed": True, "source": "benchmark/smoke_cases.json"},
            "hard_negative": False,
            "supplied_evidence": None,
            "tags": ["smoke"],
        })
    return out


def building_cases() -> list[dict]:
    path = ROOT / "benchmark" / "runs" / "building_query_results.json"
    out = []
    for c in json.loads(path.read_text(encoding="utf-8"))["cases"]:
        same = c["expected_same_building"]
        loc = "clough" if same else "tech-tower"
        out.append({
            "case_id": f"GT-{c['case_id']}",
            "dataset": "gatech_pilot",
            "location_id": loc,
            "split": split_for(loc),
            "query_type": "cross_view_matching",
            "difficulty": c["difficulty"],
            "answerable": True,
            "expected_action": "verify_then_answer",
            "question": c["question"],
            "images": [c["image_a"], c["image_b"]],
            "reference": {"answer": c["reference_answer"], "required_points": list(c["reference_landmarks"].values()),
                          "unsupported_claims_to_avoid": []},
            "labels": {"author": "human", "reviewed": True, "source": "benchmark/runs/building_query_results.json"},
            "hard_negative": not same,
            "supplied_evidence": None,
            "tags": ["cross_view_adversarial"],
        })
    return out


def main() -> int:
    cases = pilot_cases() + smoke_cases() + building_cases()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"schema_version": "2.0", "dataset": "gatech_pilot", "cases": cases}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
