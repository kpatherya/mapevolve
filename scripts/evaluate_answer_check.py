"""Does the answer checker spot bad answers? Decide the cascade with evidence.

    python -m scripts.evaluate_answer_check                 # Jev (image-aware) if TYPESAFE_API_KEY, else heuristic
    python -m scripts.evaluate_answer_check --checker heuristic

Runs the checker on every graded answer that has its text (pilot + live runs)
and compares its risk with the grades:
  * AUC for spotting answers with an unsupported claim
  * AUC for spotting bad answers (score <= 2)
  * precision / recall / F1 at thresholds 0.2-0.8
The threshold is chosen on dev cases only (best F1 for unsupported claims) and
then reported on test, so the cascade threshold isn't fit to the cases it is
judged on. Jev results are cached in results/answer_checks/ (one call per
answer, images attached).

Writes results/answer_checks/eval_<checker>.{json,md}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Optional

from dotenv import load_dotenv

from jev_router.answer_check import AnswerChecker, heuristic_check
from jev_router.classify_cache import classify_cases
from jev_router.dataset import ROOT, load_all
from jev_router.graded import load_graded

OUT = ROOT / "results" / "answer_checks"
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def auc(scores: list[float], labels: list[bool]) -> Optional[float]:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return round(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)), 3)


def at_threshold(scores: list[float], labels: list[bool], t: float) -> dict[str, Any]:
    tp = sum(s >= t and y for s, y in zip(scores, labels))
    fp = sum(s >= t and not y for s, y in zip(scores, labels))
    fn = sum(s < t and y for s, y in zip(scores, labels))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
    return {"threshold": t, "flagged": tp + fp, "precision": None if prec is None else round(prec, 3),
            "recall": None if rec is None else round(rec, 3), "f1": round(f1, 3)}


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checker", choices=("auto", "jev", "heuristic"), default="auto")
    parser.add_argument("--classifier", choices=("auto", "jev", "heuristic"), default="auto")
    args = parser.parse_args()
    has_key = bool(os.environ.get("TYPESAFE_API_KEY"))
    checker_mode = args.checker if args.checker != "auto" else ("jev" if has_key else "heuristic")
    cls_mode = args.classifier if args.classifier != "auto" else ("jev" if has_key else "heuristic")

    cases = {c.case_id: c for c in load_all()}
    rows = [g for g in load_graded(keep_repeats=True) if g.case_id in cases and g.response_text]
    if not rows:
        print("No graded answers with text found.", file=sys.stderr)
        return 1
    cls = classify_cases([cases[c] for c in sorted({g.case_id for g in rows})], cls_mode)
    checker = AnswerChecker(use_jev=(checker_mode == "jev"))

    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / f"{checker_mode}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            cache[d["key"]] = d
    results = []
    with cache_path.open("a", encoding="utf-8") as fh:
        for g in rows:
            c = cases[g.case_id]
            key = hashlib.sha256(f"{g.case_id}|{g.model_key}|{g.run_id}|{g.response_text}".encode()).hexdigest()[:16]
            if key in cache:
                risk, source = cache[key]["risk"], cache[key]["source"]
            elif checker_mode == "heuristic":
                r = heuristic_check(g.response_text, c.question, cls[g.case_id], c.supplied_evidence)
                risk, source = r.risk, r.source
            else:
                r = checker.check(g.response_text, c.question, c.image_paths(), cls[g.case_id], c.supplied_evidence)
                risk, source = r.risk, r.source
                if r.source == "jev_answer_check":  # never cache a fallback as a Jev result
                    fh.write(json.dumps({"key": key, "case_id": g.case_id, "model": g.model_key, "risk": risk,
                                         "source": source, "signals": r.signals}) + "\n")
            results.append({"case_id": g.case_id, "split": c.split, "model": g.model_key, "risk": risk,
                            "source": source, "unsupported": g.unsupported, "bad": g.score <= 2, "score": g.score})

    def block(rs):
        risks = [r["risk"] for r in rs]
        uns = [r["unsupported"] for r in rs]
        bad = [r["bad"] for r in rs]
        return {"n": len(rs), "n_unsupported": sum(uns), "n_bad": sum(bad),
                "auc_unsupported": auc(risks, uns), "auc_bad": auc(risks, bad),
                "thresholds": [at_threshold(risks, uns, t) for t in THRESHOLDS]}

    dev = [r for r in results if r["split"] == "dev"]
    test = [r for r in results if r["split"] == "test"]
    dev_block = block(dev) if dev else None
    chosen = max(dev_block["thresholds"], key=lambda t: (t["f1"], -t["threshold"]))["threshold"] if dev_block else None
    report = {"checker": checker_mode, "sources": sorted({r["source"] for r in results}), "all": block(results),
              "dev": dev_block, "test": block(test) if test else None, "chosen_threshold_on_dev": chosen,
              "test_at_chosen": at_threshold([r["risk"] for r in test], [r["unsupported"] for r in test], chosen)
              if test and chosen is not None else None}
    a = report["all"]
    good = (a["auc_unsupported"] or 0) >= 0.75
    report["verdict"] = ("strong enough to try the cascade (set cascade.enabled and risk_threshold)" if good else
                         "too weak for the cascade; keep cascade.enabled: false")
    (OUT / f"eval_{checker_mode}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Answer checker evaluation — {checker_mode} ({a['n']} graded answers)", "",
             f"**Verdict: {report['verdict']}.** Rule of thumb: AUC >= 0.75 on unsupported claims.", "",
             "| Slice | n | Unsupported | Bad (score ≤ 2) | AUC unsupported | AUC bad |", "| --- | --- | --- | --- | --- | --- |"]
    for name in ("all", "dev", "test"):
        b = report[name]
        if b:
            lines.append(f"| {name} | {b['n']} | {b['n_unsupported']} | {b['n_bad']} | {b['auc_unsupported']} | {b['auc_bad']} |")
    lines += ["", f"Threshold chosen on dev (best F1 for unsupported claims): {chosen}", ""]
    if report["test_at_chosen"]:
        t = report["test_at_chosen"]
        lines.append(f"On test at {chosen}: flagged {t['flagged']}, precision {t['precision']}, recall {t['recall']}, F1 {t['f1']}.")
    lines += ["", "| Threshold | Flagged | Precision | Recall | F1 (all) |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {t['threshold']} | {t['flagged']} | {t['precision']} | {t['recall']} | {t['f1']} |" for t in a["thresholds"]]
    (OUT / f"eval_{checker_mode}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
