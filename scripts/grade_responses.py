"""Grade live responses with an LLM judge, or import human grades.

    # LLM judge (text-only; grades against the manifest reference)
    python -m scripts.grade_responses judge results/live_runs/<run>.jsonl --backend openai --model gpt-5.1

    # Human review round-trip
    python -m scripts.grade_responses export results/live_runs/<run>.jsonl   # -> results/review/<run>.csv
    python -m scripts.grade_responses import results/review/<run>.csv        # filled human_score column

Graded rows go to results/graded/<run>.<source>.jsonl and are picked up by
train_quality_model and evaluate_offline. Human scores override judge scores
for the same (case, model). The judge never sees the images, so its grades
are only as good as the reference; prefer human review for final numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from jev_router.dataset import ROOT, load_all
from jev_router.graded import GRADED_DIR, Graded, append_graded


JUDGE_PROMPT = ROOT / "config" / "prompts" / "judge_v2.txt"
REVIEW_DIR = ROOT / "results" / "review"


def parse_verdict(text: str) -> dict:
    """Extract the judge's JSON object (tolerates code fences and surrounding prose)."""
    import re

    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and "score" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            continue
    return {}


def _lines(items) -> str:
    return "\n".join(f"- {x}" for x in items) or "- (none)"


def render(case, answer: str) -> str:
    ref = case.reference
    text = JUDGE_PROMPT.read_text(encoding="utf-8")
    for token, value in {
        "<<<QUESTION>>>": case.question, "<<<EVIDENCE>>>": case.supplied_evidence or "(none)",
        "<<<EXPECTED_ACTION>>>": case.expected_action, "<<<ANSWERABLE>>>": str(case.answerable),
        "<<<REFERENCE>>>": ref["answer"], "<<<REQUIRED>>>": _lines(ref.get("required_points", [])),
        "<<<CONFOUNDERS>>>": _lines(ref.get("acceptable_confounders", [])),
        "<<<AVOID>>>": _lines(ref.get("unsupported_claims_to_avoid", [])), "<<<ANSWER>>>": answer,
    }.items():
        text = text.replace(token, value)
    return text


def judge(args) -> int:
    from langchain_core.messages import HumanMessage

    if args.backend == "openai":
        from langchain_openai import ChatOpenAI
        chat = ChatOpenAI(model=args.model, temperature=0, reasoning_effort="none", timeout=120, max_retries=2)
    else:
        from langchain_openrouter import ChatOpenRouter
        chat = ChatOpenRouter(model=args.model, temperature=0, timeout=120, max_retries=2)
    cases = {c.case_id: c for c in load_all()}
    rows = [json.loads(l) for l in Path(args.path).read_text(encoding="utf-8").splitlines()]
    out = GRADED_DIR / f"{Path(args.path).stem}.judge.jsonl"
    done = set()
    if out.exists():
        done = {(json.loads(l)["case_id"], json.loads(l)["model_key"]) for l in out.read_text().splitlines()}
    graded = []
    for r in rows:
        if r["status"] != "ok" or (r["case_id"], r["model_key"]) in done or r["case_id"] not in cases:
            continue
        reply = chat.invoke([HumanMessage(content=render(cases[r["case_id"]], r["answer"]))])
        verdict = parse_verdict(reply.content if isinstance(reply.content, str) else str(reply.content))
        try:
            score = max(0.0, min(4.0, float(verdict["score"])))
        except (KeyError, TypeError, ValueError):
            print(f"unparseable judge reply for {r['case_id']}/{r['model_key']}", file=sys.stderr)
            continue
        unsupported = bool(verdict.get("unsupported_claim"))
        if unsupported:
            score = min(score, 3.0)
        label = verdict.get("label") or ("correct" if score >= 4 else "partial" if score >= 2 else "incorrect")
        graded.append(Graded(r["case_id"], r["model_key"], score, label, unsupported, f"llm_judge:{args.model}",
                             r.get("latency_seconds"), r.get("cost_usd"), r.get("input_tokens"),
                             r.get("output_tokens"), r.get("answer") or ""))
        print(f"{r['case_id']:<26} {r['model_key']:<24} {score} {label}{' (unsupported)' if unsupported else ''}")
    append_graded(out, graded)
    print(f"wrote {len(graded)} grades to {out.relative_to(ROOT)}")
    return 0


def export(args) -> int:
    cases = {c.case_id: c for c in load_all()}
    rows = [json.loads(l) for l in Path(args.path).read_text(encoding="utf-8").splitlines()]
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = REVIEW_DIR / f"{Path(args.path).stem}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "model_key", "question", "reference_answer", "model_answer", "latency_seconds",
                    "cost_usd", "human_score", "human_label", "human_unsupported_claim", "human_notes"])
        for r in rows:
            if r["status"] == "ok" and r["case_id"] in cases:
                c = cases[r["case_id"]]
                w.writerow([r["case_id"], r["model_key"], c.question, c.reference["answer"], r["answer"],
                            r.get("latency_seconds"), r.get("cost_usd"), "", "", "", ""])
    print(f"wrote {out.relative_to(ROOT)}; fill human_score (0-4), human_label, human_unsupported_claim (true/false)")
    return 0


def import_human(args) -> int:
    graded = []
    with Path(args.path).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("human_score", "").strip():
                continue
            score = float(r["human_score"])
            graded.append(Graded(r["case_id"], r["model_key"], score,
                                 r.get("human_label") or ("correct" if score >= 4 else "partial" if score >= 2 else "incorrect"),
                                 r.get("human_unsupported_claim", "").strip().lower() in {"true", "1", "yes", "y"},
                                 "human", float(r["latency_seconds"]) if r.get("latency_seconds") else None,
                                 float(r["cost_usd"]) if r.get("cost_usd") else None, response_text=r.get("model_answer", "")))
    out = GRADED_DIR / f"{Path(args.path).stem}.human.jsonl"
    append_graded(out, graded)
    print(f"imported {len(graded)} human grades -> {out.relative_to(ROOT)}")
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("path")
    j.add_argument("--backend", choices=("openai", "openrouter"), default="openai")
    j.add_argument("--model", required=True, help="Judge model ID, e.g. gpt-5.1")
    e = sub.add_parser("export")
    e.add_argument("path")
    i = sub.add_parser("import")
    i.add_argument("path")
    args = parser.parse_args()
    if args.cmd == "judge":
        var = "OPENAI_API_KEY" if args.backend == "openai" else "OPENROUTER_API_KEY"
        if not os.environ.get(var):
            print(f"error: {var} is required", file=sys.stderr)
            return 2
        return judge(args)
    return export(args) if args.cmd == "export" else import_human(args)


if __name__ == "__main__":
    sys.exit(main())
