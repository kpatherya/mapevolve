"""Measure the classifier (Jev or the fallback) against manifest labels.

    python -m scripts.evaluate_classifier                     # auto: Jev if TYPESAFE_API_KEY, else heuristic
    python -m scripts.evaluate_classifier --classifier jev --image-aware --split dev

This isolates the classification stage from routing: if a route is bad, this
report says whether the query type / difficulty / action flags were wrong.

Only `routing_inputs(case)` (question, images, supplied evidence) are passed
to the classifier. Classifications are cached in results/classifications/
keyed by an input hash, so Jev is called once per case and configuration.

Outputs: results/classifier_eval_<tag>.{json,md,csv}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any

from jev_router.actions import decide_action
from jev_router.classify_cache import classify_cases
from jev_router.dataset import ACTIONS
from jev_router.registry import load_routing_config
from jev_router.dataset import ROOT, Case, load_all
from jev_router.schemas import DIFFICULTIES, QUERY_TYPES, QueryClassification

# ------------------------------------------------------------------ metrics
def macro_f1(pairs: list[tuple[str, str]], labels) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in pairs if t == lab and p == lab)
        fp = sum(1 for t, p in pairs if t != lab and p == lab)
        fn = sum(1 for t, p in pairs if t == lab and p != lab)
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return round(sum(f1s) / len(f1s), 4) if f1s else 0.0


def confusion(pairs: list[tuple[str, str]], labels) -> dict[str, dict[str, int]]:
    m = {t: {p: 0 for p in labels} for t in labels}
    for t, p in pairs:
        if t in m and p in m[t]:
            m[t][p] += 1
    return m


def ece(conf_correct: list[tuple[float, bool]], bins: int = 10) -> float:
    if not conf_correct:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, k) for c, k in conf_correct if lo < c <= hi or (b == 0 and c == 0)]
        if bucket:
            acc = sum(k for _, k in bucket) / len(bucket)
            avg = sum(c for c, _ in bucket) / len(bucket)
            total += len(bucket) / len(conf_correct) * abs(acc - avg)
    return round(total, 4)


def brier(prob_rows: list[tuple[dict[str, float], str]], labels) -> float:
    if not prob_rows:
        return 0.0
    s = sum(sum((probs.get(l, 0.0) - (1.0 if l == truth else 0.0)) ** 2 for l in labels) for probs, truth in prob_rows)
    return round(s / len(prob_rows), 4)


def binary(rows: list[tuple[float, bool]]) -> dict[str, Any]:
    tp = sum(1 for p, y in rows if p >= 0.5 and y)
    fp = sum(1 for p, y in rows if p >= 0.5 and not y)
    fn = sum(1 for p, y in rows if p < 0.5 and y)
    tn = sum(1 for p, y in rows if p < 0.5 and not y)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {"n": len(rows), "positives": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "accuracy": round((tp + tn) / len(rows), 4) if rows else None,
            "brier": round(sum((p - y) ** 2 for p, y in rows) / len(rows), 4) if rows else None}


_CFG = load_routing_config()


def evaluate(cases: list[Case], cls: dict[str, QueryClassification]) -> dict[str, Any]:
    act = [(c.expected_action, decide_action(cls[c.case_id], _CFG, c.supplied_evidence).action) for c in cases]
    qt = [(c.query_type, cls[c.case_id].query_type.label) for c in cases]
    df = [(c.difficulty, cls[c.case_id].difficulty.label) for c in cases]
    idx = {d: i for i, d in enumerate(DIFFICULTIES)}
    return {
        "n": len(cases),
        "query_type": {
            "accuracy": round(sum(t == p for t, p in qt) / len(qt), 4),
            "macro_f1": macro_f1(qt, QUERY_TYPES),
            "ece_top_label": ece([(max(cls[c.case_id].query_type.probabilities.values()),
                                   c.query_type == cls[c.case_id].query_type.label) for c in cases]),
            "brier": brier([(cls[c.case_id].query_type.probabilities, c.query_type) for c in cases], QUERY_TYPES),
            "confusion": confusion(qt, QUERY_TYPES),
        },
        "difficulty": {
            "accuracy": round(sum(t == p for t, p in df) / len(df), 4),
            "within_one": round(sum(abs(idx[t] - idx[p]) <= 1 for t, p in df) / len(df), 4),
            "mean_abs_error": round(sum(abs(idx[t] - idx[p]) for t, p in df) / len(df), 4),
            "brier": brier([(cls[c.case_id].difficulty.probabilities, c.difficulty) for c in cases], DIFFICULTIES),
            "confusion": confusion(df, DIFFICULTIES),
        },
        "action": {
            "accuracy": round(sum(t == p for t, p in act) / len(act), 4),
            "macro_f1": macro_f1(act, ACTIONS),
            "confusion": confusion(act, ACTIONS),
        },
        "flags": {
            "needs_external_tools_vs_use_tools": binary(
                [(cls[c.case_id].needs_external_tools, c.expected_action == "use_tools") for c in cases]),
            "needs_pair_verification_vs_verify": binary(
                [(cls[c.case_id].needs_pair_verification, c.expected_action == "verify_then_answer") for c in cases]),
            "answerable_from_inputs_vs_answerable": binary(
                [(cls[c.case_id].answerable_from_inputs, c.answerable) for c in cases]),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# Classifier evaluation — {report['meta']['classifier']}"
             f"{' (image-aware)' if report['meta']['image_aware'] else ''}", "",
             f"Sources used: {report['meta']['sources']} · cases: {report['overall']['n']} · "
             f"labels: {report['meta']['label_review']}", ""]
    lines += ["| Slice | n | Type acc | Type macro-F1 | Type ECE | Diff acc | Diff within-1 | Action acc | Tools P/R | Verify P/R | Answerable acc |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in [("overall", report["overall"]), *report["slices"].items()]:
        f = r["flags"]
        pr = lambda b: f"{b['precision']}/{b['recall']}"
        lines.append(f"| {name} | {r['n']} | {r['query_type']['accuracy']} | {r['query_type']['macro_f1']} | "
                     f"{r['query_type']['ece_top_label']} | {r['difficulty']['accuracy']} | {r['difficulty']['within_one']} | "
                     f"{r['action']['accuracy']} | "
                     f"{pr(f['needs_external_tools_vs_use_tools'])} | {pr(f['needs_pair_verification_vs_verify'])} | "
                     f"{f['answerable_from_inputs_vs_answerable']['accuracy']} |")
    lines += ["", "## Difficulty confusion (rows = label, cols = predicted)", "",
              "| | " + " | ".join(DIFFICULTIES) + " |", "| --- |" + " --- |" * len(DIFFICULTIES)]
    for t, row in report["overall"]["difficulty"]["confusion"].items():
        lines.append(f"| {t} | " + " | ".join(str(row[p]) for p in DIFFICULTIES) + " |")
    lines += ["", "## Query-type confusion", "", "| | " + " | ".join(q[:12] for q in QUERY_TYPES) + " |",
              "| --- |" + " --- |" * len(QUERY_TYPES)]
    for t, row in report["overall"]["query_type"]["confusion"].items():
        lines.append(f"| {t} | " + " | ".join(str(row[p]) for p in QUERY_TYPES) + " |")
    lines += ["", "## Action confusion (rows = expected, cols = chosen)", "",
              "| | " + " | ".join(ACTIONS) + " |", "| --- |" + " --- |" * len(ACTIONS)]
    for t, row in report["overall"]["action"]["confusion"].items():
        lines.append(f"| {t} | " + " | ".join(str(row[p]) for p in ACTIONS) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", choices=("auto", "jev", "heuristic"), default="auto")
    parser.add_argument("--image-aware", action="store_true")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--dataset", default=None, help="Only this dataset (e.g. demo_world)")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached classifications")
    args = parser.parse_args()

    mode = args.classifier
    if mode == "auto":
        mode = "jev" if os.environ.get("TYPESAFE_API_KEY") else "heuristic"
    cases = [c for c in load_all() if (args.split == "all" or c.split == args.split)
             and (args.dataset is None or c.dataset == args.dataset)]
    cls = classify_cases(cases, mode, args.image_aware, args.refresh)

    slices: dict[str, list[Case]] = defaultdict(list)
    for c in cases:
        slices[f"dataset={c.dataset}"].append(c)
        slices[f"split={c.split}"].append(c)
    report = {
        "meta": {"classifier": mode, "image_aware": args.image_aware, "split": args.split,
                 "sources": dict(Counter(v.source for v in cls.values())),
                 "label_review": dict(Counter("reviewed" if c.labels.get("reviewed") else "unreviewed" for c in cases))},
        "overall": evaluate(cases, cls),
        "slices": {k: evaluate(v, cls) for k, v in sorted(slices.items())},
    }
    tag = f"{mode}{'_img' if args.image_aware else ''}"
    out = ROOT / "results"
    (out / f"classifier_eval_{tag}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = markdown(report)
    (out / f"classifier_eval_{tag}.md").write_text(md, encoding="utf-8")
    with (out / f"classifier_eval_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "dataset", "split", "label_type", "pred_type", "label_difficulty", "pred_difficulty",
                    "expected_action", "p_tools", "p_verify", "answerable", "p_answerable", "source"])
        for c in cases:
            k = cls[c.case_id]
            w.writerow([c.case_id, c.dataset, c.split, c.query_type, k.query_type.label, c.difficulty,
                        k.difficulty.label, c.expected_action, k.needs_external_tools, k.needs_pair_verification,
                        c.answerable, k.answerable_from_inputs, k.source])
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
