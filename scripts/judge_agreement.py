"""How far can the LLM judge be trusted? Compare judge and human grades.

    python -m scripts.judge_agreement

Joins every judge grade and human grade in results/graded/ on
(case, model, run). Reports exact and within-one agreement, mean bias
(judge - human), quadratic-weighted kappa on the 0-4 score, agreement and
Cohen's kappa on the unsupported-claim flag, a score confusion matrix, a
per-query-type breakdown, and the largest disagreements to read by hand.

Rule of thumb: weighted kappa >= 0.6 and unsupported-flag kappa >= 0.5 means
the judge is usable for routing data; below that, prefer human grades
(they already override judge grades automatically).

Writes results/grading/agreement.{json,md}.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import mean

from jev_router.dataset import ROOT, load_all
from jev_router.graded import GRADED_DIR

OUT = ROOT / "results" / "grading"


def weighted_kappa(a: list[int], b: list[int], k: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal scores 0..k-1."""
    n = len(a)
    if n == 0:
        return float("nan")
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    ha = [sum(obs[i]) for i in range(k)]
    hb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * obs[i][j]
            den += w * ha[i] * hb[j] / n
    return 1 - num / den if den else 1.0


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_pairs():
    judge, human = {}, {}
    for path in sorted(GRADED_DIR.glob("*.jsonl")) if GRADED_DIR.exists() else []:
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            key = (d["case_id"], d["model_key"], d.get("run_id"))
            (human if d["score_source"] == "human" else judge if d["score_source"].startswith("llm_judge") else {})[key] = d
    return [(k, judge[k], human[k]) for k in sorted(set(judge) & set(human))]


def summarize(pairs) -> dict:
    js = [round(j["score"]) for _, j, _ in pairs]
    hs = [round(h["score"]) for _, _, h in pairs]
    ju = [bool(j["unsupported"]) for _, j, _ in pairs]
    hu = [bool(h["unsupported"]) for _, _, h in pairs]
    return {
        "n": len(pairs),
        "exact": round(mean(a == b for a, b in zip(js, hs)), 3),
        "within_one": round(mean(abs(a - b) <= 1 for a, b in zip(js, hs)), 3),
        "mean_abs_diff": round(mean(abs(a - b) for a, b in zip(js, hs)), 3),
        "bias_judge_minus_human": round(mean(a - b for a, b in zip(js, hs)), 3),
        "weighted_kappa": round(weighted_kappa(js, hs), 3),
        "unsupported_agreement": round(mean(a == b for a, b in zip(ju, hu)), 3),
        "unsupported_kappa": round(cohen_kappa(ju, hu), 3),
    }


def main() -> int:
    pairs = load_pairs()
    if not pairs:
        print("No (case, model, run) has both a judge grade and a human grade yet.\n"
              "Run: grade_responses judge <run> ; grade_responses export <run> --sample 0.2 ; fill it ; "
              "grade_responses import <csv>", file=sys.stderr)
        return 1
    cases = {c.case_id: c for c in load_all()}
    overall = summarize(pairs)
    by_type = defaultdict(list)
    for p in pairs:
        c = cases.get(p[0][0])
        by_type[c.query_type if c else "unknown"].append(p)
    confusion = [[0] * 5 for _ in range(5)]
    for _, j, h in pairs:
        confusion[round(h["score"])][round(j["score"])] += 1
    worst = sorted(pairs, key=lambda p: -abs(p[1]["score"] - p[2]["score"]))[:10]
    report = {"overall": overall, "by_query_type": {t: summarize(v) for t, v in sorted(by_type.items())},
              "confusion_rows_human_cols_judge": confusion,
              "largest_disagreements": [{"case_id": k[0], "model": k[1], "run_id": k[2], "judge": j["score"],
                                         "human": h["score"], "answer": (j.get("response_text") or "")[:300]}
                                        for k, j, h in worst]}
    usable = overall["weighted_kappa"] >= 0.6 and overall["unsupported_kappa"] >= 0.5
    report["verdict"] = ("judge usable for routing data" if usable
                         else "judge NOT reliable enough; prefer human grades for these cells")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "agreement.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    o = overall
    lines = [f"# Judge vs human agreement ({o['n']} answers graded by both)", "",
             f"**Verdict: {report['verdict']}** (weighted kappa {o['weighted_kappa']}, unsupported-flag kappa "
             f"{o['unsupported_kappa']}; thresholds 0.6 / 0.5)", "",
             "| Slice | n | Exact | Within 1 | Bias (judge−human) | Weighted κ | Unsupported agree | Unsupported κ |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in [("overall", o), *report["by_query_type"].items()]:
        lines.append(f"| {name} | {r['n']} | {r['exact']} | {r['within_one']} | {r['bias_judge_minus_human']} | "
                     f"{r['weighted_kappa']} | {r['unsupported_agreement']} | {r['unsupported_kappa']} |")
    lines += ["", "Score confusion (rows = human, columns = judge):", "", "| | 0 | 1 | 2 | 3 | 4 |",
              "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {i} | " + " | ".join(str(x) for x in row) + " |" for i, row in enumerate(confusion)]
    lines += ["", "Largest disagreements:", ""]
    lines += [f"- {d['case_id']} / {d['model']}: judge {d['judge']}, human {d['human']}"
              for d in report["largest_disagreements"]]
    (OUT / "agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
