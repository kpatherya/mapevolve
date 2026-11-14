"""Walk one case through the whole system and print every step. For learning.

    python -m scripts.explain_case                         # a default demo case
    python -m scripts.explain_case W-C-co-day-night        # any case ID
    python -m scripts.explain_case W-C-co-day-night --mode economy
    python -m scripts.explain_case --list                  # show case IDs

Steps printed:
  1. the input (question, images, evidence)
  2. what is sent to Jev (the state) and the seven questions
  3. Jev's answers with their probabilities
  4. the action rule that turns those answers into an action
  5. the router's table: every model's expected quality, cost and whether it qualifies
  6. the model it picks, and why
  7. (if graded) what each model actually scored on this case

Uses the cached Jev classification when there is one (no API call, no cost).
Without a TYPESAFE_API_KEY it falls back to the keyword heuristic and says so.
Calls no VLM.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from jev_router.actions import action_thresholds, decide_action, verify_probability
from jev_router.classifier import build_state
from jev_router.classify_cache import classify_cases, resolve_image_aware
from jev_router.dataset import load_all, routing_inputs
from jev_router.graded import load_graded
from jev_router.policy import RoutingPolicy
from jev_router.registry import ModelRegistry, load_routing_config

DEFAULT_CASE = "W-T-bt-same-day"
LINE = "-" * 78


def bar(p: float, width: int = 20) -> str:
    return "#" * round(p * width) + "." * (width - round(p * width))


def section(n: int, title: str) -> None:
    print(f"\n{LINE}\n{n}. {title}\n{LINE}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_id", nargs="?", default=DEFAULT_CASE)
    parser.add_argument("--mode", choices=("economy", "balanced", "accuracy"), default="balanced")
    parser.add_argument("--list", action="store_true", help="List case IDs and exit")
    args = parser.parse_args()

    cases = {c.case_id: c for c in load_all()}
    if args.list:
        for c in cases.values():
            print(f"{c.case_id:<50} {c.query_type:<24} {c.difficulty:<9} {c.question[:60]}")
        return 0
    case = cases.get(args.case_id)
    if case is None:
        print(f"unknown case {args.case_id!r}; use --list", file=sys.stderr)
        return 2

    cfg = load_routing_config()
    mode = "jev" if os.environ.get("TYPESAFE_API_KEY") else "heuristic"
    image_aware = resolve_image_aware(None, mode)

    section(1, "Input (what a user would give the system)")
    inp = routing_inputs(case)
    print(f"Question : {inp['question']}")
    for p in case.images:
        print(f"Image    : {p}")
    print(f"Evidence : {inp['supplied_evidence'] or '(none)'}")
    print("(The reference answer, labels and expected action are hidden from the router.)")

    section(2, "What is sent to Jev")
    state = build_state(inp["question"], inp["images"], None, image_aware=image_aware,
                        evidence=inp["supplied_evidence"])
    print("A 'state' (facts about the query) plus seven questions, in ONE call.")
    print(f"State keys: {', '.join(k for k in state if k != 'images')}"
          f"{' + the images themselves' if image_aware else ''}")
    pair = state.get("pair_features")
    if pair:
        kp = pair.get("keypoints") or {}
        print(f"  pixel check computed locally: viewpoint {pair.get('viewpoint_alignment')}, "
              f"{kp.get('inliers')} matching keypoints, colour similarity {pair.get('histogram_similarity')}")
    print("Questions (written by us in jev_router/classifier.py, build_questions):")
    for name, kind in [("query_type", "Choice: one of 5 classes"), ("difficulty", "Score: easy/moderate/hard"),
                       ("visual_expertise", "Choice: one of several skills"),
                       ("needs_image_retrieval", "yes/no"), ("needs_pair_verification", "yes/no"),
                       ("needs_external_tools", "yes/no"), ("answerable_from_inputs", "yes/no")]:
        print(f"  - {name:<24} {kind}")

    section(3, "Jev's answers (probabilities, not just labels)")
    cls = classify_cases([case], mode, image_aware)[case.case_id]
    print(f"Source: {cls.source}{' (keyword fallback: no TypeSafe key)' if cls.source != 'jev' else ''}"
          f"{', images sent' if image_aware and cls.source == 'jev' else ''}")
    for name in ("query_type", "difficulty", "visual_expertise"):
        dist = getattr(cls, name)
        print(f"\n  {name} -> {dist.label}")
        for label, p in sorted(dist.probabilities.items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {label:<34} {bar(p)} {p:.2f}")
    print()
    for name in ("needs_image_retrieval", "needs_pair_verification", "needs_external_tools", "answerable_from_inputs"):
        p = getattr(cls, name)
        print(f"  {name:<24} P(yes) {bar(p)} {p:.2f}")

    section(4, "Action rule (fixed code in jev_router/actions.py, not Jev)")
    th = action_thresholds(cfg, cls.source)
    pv, pv_name = verify_probability(cls, th)
    print("Checked in this order; the first rule that fires wins:")
    print(f"  decline  if P(answerable) < {th['decline_if_answerable_below']} and P(tools) < {th['tools_threshold']}"
          f"   [{cls.answerable_from_inputs:.2f}, {cls.needs_external_tools:.2f}]")
    print(f"  tools    if P(tools) >= {th['tools_threshold']}   [{cls.needs_external_tools:.2f}]")
    print(f"  retrieve if P(retrieval) >= {th['retrieval_threshold']}   [{cls.needs_image_retrieval:.2f}]")
    print(f"  verify   if 2 images and {pv_name} >= {th['verify_threshold']}   [{pv:.2f}]")
    print("  answer   otherwise")
    action = decide_action(cls, cfg, inp["supplied_evidence"])
    print(f"\n=> action: {action.action}")
    for r in action.reasons:
        print(f"   {r}")
    if action.prompt_note:
        print(f"   Note added to the model's prompt: {action.prompt_note}")
    print(f"   (expected action in the labels: {case.expected_action})")

    section(5, f"Router table, mode = {args.mode} (fixed code in jev_router/policy.py)")
    registry = ModelRegistry.load(routing=cfg)
    decision = RoutingPolicy(registry, cfg, use_quality_model=False).route(cls, len(case.images), args.mode)
    m = cfg["modes"][args.mode]
    print(f"Mode settings: { {k: v for k, v in m.items() if k != 'description'} }")
    print("Expected quality = this model's past scores on this class x difficulty, weighted by Jev's")
    print("probabilities, pulled toward its overall score when there are few cases (0-1 scale).\n")
    print(f"  {'model':<24}{'exp. quality':>13}{'cost/query':>12}{'latency s':>10}  status")
    for c in sorted(decision.candidates, key=lambda c: -(c.adjusted_quality or -1)):
        q = f"{c.adjusted_quality:.3f}" if c.adjusted_quality is not None else "-"
        cost = f"${c.est_cost_usd:.5f}" if c.est_cost_usd is not None else "-"
        lat = f"{c.est_latency_seconds:.1f}" if c.est_latency_seconds is not None else "-"
        status = ("PICKED" if c.key == decision.selected_key else "ok") if c.eligible else \
            "excluded: " + "; ".join(c.exclusion_reasons)[:60]
        print(f"  {c.key:<24}{q:>13}{cost:>12}{lat:>10}  {status}")

    section(6, "Decision")
    print(f"=> {decision.selected_key} ({decision.selected_model_id})")
    for r in decision.rationale:
        print(f"   {r}")
    for w in decision.warnings:
        print(f"   warning: {w}")

    graded = [g for g in load_graded(keep_repeats=True) if g.case_id == case.case_id]
    if graded:
        section(7, "What each model actually scored on this case (0-4, from grading)")
        best: dict[str, float] = {}
        for g in graded:
            best[g.model_key] = max(best.get(g.model_key, 0), g.score)
        for k, s in sorted(best.items(), key=lambda kv: -kv[1]):
            mark = "  <= router's pick" if k == decision.selected_key else ""
            print(f"  {k:<24} {s:.0f}{mark}")
        print("\n(Routing never sees these scores for the case it routes in the held-out evaluation.)")
    print(f"\nReference answer: {case.reference['answer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
