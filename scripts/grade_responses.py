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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

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
        chat = ChatOpenRouter(model=args.model, temperature=0, timeout=120_000, max_retries=2)  # ms
    cases = {c.case_id: c for c in load_all()}
    rows = [json.loads(l) for l in Path(args.path).read_text(encoding="utf-8").splitlines()]
    out = GRADED_DIR / f"{Path(args.path).stem}.judge.jsonl"
    done = set()
    if out.exists():
        done = {(d["case_id"], d["model_key"], d.get("run_id"))
                for d in map(json.loads, out.read_text().splitlines())}
    unreviewed = sorted({r["case_id"] for r in rows if r["case_id"] in cases
                         and not cases[r["case_id"]].labels.get("reviewed")})
    if unreviewed:
        print(f"WARNING: {len(unreviewed)} case(s) have unreviewed reference answers; their grades are only as good "
              f"as the draft: {', '.join(unreviewed[:8])}{' ...' if len(unreviewed) > 8 else ''}", file=sys.stderr)
        if args.reviewed_only:
            rows = [r for r in rows if r["case_id"] not in unreviewed]
    todo = [r for r in rows if r["status"] == "ok" and r["case_id"] in cases
            and (r["case_id"], r["model_key"], r.get("run_id")) not in done]
    todo = list({(r["case_id"], r["model_key"], r.get("run_id")): r for r in todo}.values())  # one per answer

    def grade_one(r: dict) -> Optional[Graded]:
        reply = chat.invoke([HumanMessage(content=render(cases[r["case_id"]], r["answer"]))])
        verdict = parse_verdict(reply.content if isinstance(reply.content, str) else str(reply.content))
        try:
            score = max(0.0, min(4.0, float(verdict["score"])))
        except (KeyError, TypeError, ValueError):
            print(f"unparseable judge reply for {r['case_id']}/{r['model_key']}", file=sys.stderr)
            return None
        unsupported = bool(verdict.get("unsupported_claim"))
        if unsupported:
            score = min(score, 3.0)
        label = verdict.get("label") or ("correct" if score >= 4 else "partial" if score >= 2 else "incorrect")
        return Graded(r["case_id"], r["model_key"], score, label, unsupported, f"llm_judge:{args.model}",
                      r.get("latency_seconds"), r.get("cost_usd"), r.get("input_tokens"),
                      r.get("output_tokens"), r.get("answer") or "", run_id=r.get("run_id"))

    print(f"judging {len(todo)} answers with {args.workers} worker(s)")
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(grade_one, r): r for r in todo}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                g = fut.result()
            except Exception as exc:  # one failed call must not lose the others
                print(f"judge call failed for {r['case_id']}/{r['model_key']}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                g = None
            if g is None:
                n_fail += 1
                continue
            append_graded(out, [g])  # saved immediately, so an interrupted run resumes
            n_ok += 1
            print(f"{g.case_id:<26} {g.model_key:<24} {g.score} {g.label}{' (unsupported)' if g.unsupported else ''}")
    print(f"wrote {n_ok} grades to {out.relative_to(ROOT)}; {n_fail} failed (re-run to retry them)")
    return 0


def stratified_sample(rows: list[dict], cases: dict, fraction: float, seed: int) -> list[dict]:
    """Sample ~fraction of answers per (query type x difficulty), at least one per
    stratum, spreading picks across models so every model gets human grades."""
    import math
    import random

    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = {}
    for r in rows:
        c = cases[r["case_id"]]
        strata.setdefault((c.query_type, c.difficulty), []).append(r)
    picked = []
    for key in sorted(strata):
        group = strata[key]
        want = max(1, math.ceil(len(group) * fraction))
        by_model: dict[str, list[dict]] = {}
        for r in group:
            by_model.setdefault(r["model_key"], []).append(r)
        for v in by_model.values():
            rng.shuffle(v)
        order = sorted(by_model)
        rng.shuffle(order)
        while len([x for x in picked if (cases[x["case_id"]].query_type, cases[x["case_id"]].difficulty) == key]) < want:
            progressed = False
            for m in order:
                if by_model[m] and len([x for x in picked if (cases[x["case_id"]].query_type,
                                                               cases[x["case_id"]].difficulty) == key]) < want:
                    picked.append(by_model[m].pop())
                    progressed = True
            if not progressed:
                break
    return picked


def export(args) -> int:
    """Blind review sheet: graders see the question, reference, images and answer,
    never the judge's score."""
    cases = {c.case_id: c for c in load_all()}
    rows = [json.loads(l) for l in Path(args.path).read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["status"] == "ok" and r["case_id"] in cases]
    if args.sample < 1.0:
        rows = stratified_sample(rows, cases, args.sample, args.seed)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.sample >= 1.0 else f".sample{int(args.sample * 100)}"
    out = REVIEW_DIR / f"{Path(args.path).stem}{suffix}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "model_key", "run_id", "query_type", "difficulty", "expected_action", "question",
                    "supplied_evidence", "images", "reference_answer", "required_points", "claims_to_avoid",
                    "model_answer", "latency_seconds", "cost_usd",
                    "human_score", "human_label", "human_unsupported_claim", "human_notes", "grader"])
        for r in rows:
            c = cases[r["case_id"]]
            ref = c.reference
            w.writerow([r["case_id"], r["model_key"], r.get("run_id"), c.query_type, c.difficulty, c.expected_action,
                        c.question, c.supplied_evidence or "", " | ".join(c.images), ref["answer"],
                        " | ".join(ref.get("required_points", [])), " | ".join(ref.get("unsupported_claims_to_avoid", [])),
                        r["answer"], r.get("latency_seconds"), r.get("cost_usd"), "", "", "", "", ""])
    print(f"wrote {out.relative_to(ROOT)} ({len(rows)} answers); fill human_score (0-4), human_label "
          f"(correct/partial/incorrect), human_unsupported_claim (true/false), grader")
    return 0


def import_human(args) -> int:
    graded = []
    with Path(args.path).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("human_score", "").strip():
                continue
            try:
                score = float(r["human_score"])
            except ValueError:
                print(f"skip {r['case_id']}/{r['model_key']}: human_score {r['human_score']!r} is not a number",
                      file=sys.stderr)
                continue
            if not 0 <= score <= 4:
                print(f"skip {r['case_id']}/{r['model_key']}: human_score {score} outside 0-4", file=sys.stderr)
                continue
            graded.append(Graded(r["case_id"], r["model_key"], score,
                                 r.get("human_label") or ("correct" if score >= 4 else "partial" if score >= 2 else "incorrect"),
                                 r.get("human_unsupported_claim", "").strip().lower() in {"true", "1", "yes", "y"},
                                 "human", float(r["latency_seconds"]) if r.get("latency_seconds") else None,
                                 float(r["cost_usd"]) if r.get("cost_usd") else None, response_text=r.get("model_answer", ""),
                                 run_id=r.get("run_id") or None, grader=r.get("grader") or None))
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
    j.add_argument("--reviewed-only", action="store_true", help="Skip cases whose references are unreviewed")
    j.add_argument("--workers", type=int, default=4, help="Parallel judge calls")
    e = sub.add_parser("export")
    e.add_argument("path")
    e.add_argument("--sample", type=float, default=1.0, help="Fraction to export, stratified (e.g. 0.2)")
    e.add_argument("--seed", type=int, default=7)
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
