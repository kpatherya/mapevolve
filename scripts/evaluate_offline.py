"""Offline evaluation of routing strategies on recorded, graded responses.

Replays graded responses instead of calling VLMs again (free, reproducible).
Today that is the Georgia Tech pilot (6 cases x 8 models); graded rows added
under results/graded/ by scripts.run_live_benchmark + scripts.grade_responses
are picked up automatically for any case that has every pool model graded.

    python -m scripts.evaluate_offline                    # Jev if TYPESAFE_API_KEY is set, else heuristic
    python -m scripts.evaluate_offline --classifier jev --image-aware

Strategies
  always_cheapest, always_strongest, fixed_task      baselines
  jev_adaptive_{economy,balanced,accuracy}           registry table
  jev_accuracy_learned                               learned predictor (forced on)
  jev_cascade_balanced                               balanced pick, heuristic answer
                                                     check on the recorded text,
                                                     escalate once to the accuracy pick
  oracle                                             best recorded score per case

Leakage controls: routing sees only routing_inputs(case); registry scores
and the learned predictor are refit leave-one-case-out (LOCO).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from jev_router.answer_check import heuristic_check
from jev_router.benchmark_data import load_responses, model_metrics
from jev_router.classifier import heuristic_classify
from jev_router.classify_cache import classify_cases
from jev_router.dataset import load_all
from jev_router.graded import load_graded
from jev_router.policy import RoutingPolicy
from jev_router.quality_model import QualityModel, TrainingRow, model_attributes
from jev_router.registry import ROOT, ModelRegistry, load_routing_config
from jev_router.schemas import MODES

RESULTS = ROOT / "results"


def est_cost(g, registry: ModelRegistry) -> tuple[Optional[float], str]:
    if g.cost_usd is not None:
        return g.cost_usd, "recorded_provider_cost"
    pricing = registry.get(g.model_key).get("pricing")
    if pricing and pricing.get("input_usd_per_mtok") is not None and g.input_tokens is not None:
        return ((g.input_tokens * pricing["input_usd_per_mtok"] + (g.output_tokens or 0) * pricing["output_usd_per_mtok"]) / 1e6,
                "estimated: recorded tokens x current reference price")
    return None, "unavailable"


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for strategy in dict.fromkeys(r["strategy"] for r in rows):
        rs = [r for r in rows if r["strategy"] == strategy and r["model"]]
        costs = [r["cost_usd"] for r in rs if r["cost_usd"] is not None]
        out.append({
            "strategy": strategy,
            "n_cases": len(rs),
            "mean_score_0_4": round(mean(r["score"] for r in rs), 4),
            "correct_rate": round(mean(r["label"] == "correct" for r in rs), 4),
            "partial_rate": round(mean(r["label"] == "partial" for r in rs), 4),
            "incorrect_rate": round(mean(r["label"] == "incorrect" for r in rs), 4),
            "unsupported_claim_rate": round(mean(bool(r["unsupported_claim"]) for r in rs), 4),
            "avg_cost_usd": round(mean(costs), 6) if costs else None,
            "n_cost_estimated": sum(str(r["cost_source"]).startswith("estimated") for r in rs),
            "avg_latency_seconds": round(mean(r["latency_seconds"] for r in rs), 3),
            "mean_regret": round(mean(r["regret"] for r in rs), 4),
            "oracle_match_rate": round(mean(r["regret"] == 0 for r in rs), 4),
            "escalations": sum(bool(r.get("escalated")) for r in rs),
            "models_used": "|".join(sorted({r["model"] for r in rs})),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", choices=("auto", "jev", "heuristic"), default="auto")
    parser.add_argument("--image-aware", action="store_true", help="Send images to Jev (experimental)")
    parser.add_argument("--budget", type=float, default=None, help="USD per query (default from config/routing.yaml)")
    parser.add_argument("--live-pool", action="store_true", help="Only route to models currently listed")
    parser.add_argument("--no-loo", action="store_true", help="Disable leave-one-case-out (in-sample; optimistic)")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    cfg = load_routing_config()
    base_registry = ModelRegistry.load(routing=cfg)
    attrs = model_attributes(base_registry.data["models"])
    manifest = {c.case_id: c for c in load_all()}
    graded = [g for g in load_graded() if g.case_id in manifest]
    by_case: dict[str, dict[str, Any]] = defaultdict(dict)
    for g in graded:
        by_case[g.case_id][g.model_key] = g
    pool = sorted({g.model_key for g in graded})
    case_ids = sorted(cid for cid, m in by_case.items() if set(m) >= set(pool))
    pilot_records = load_responses()

    mode = args.classifier if args.classifier != "auto" else ("jev" if os.environ.get("TYPESAFE_API_KEY") else "heuristic")
    cls_all = classify_cases([manifest[c] for c in case_ids], mode, args.image_aware)
    rows_for_fit = [TrainingRow(g.case_id, g.model_key, g.score, g.unsupported, cls_all[g.case_id])
                    for g in graded if g.case_id in case_ids]
    thr = cfg.get("cascade", {}).get("risk_threshold", 0.4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for cid in case_ids:
        case = manifest[cid]
        cls = cls_all[cid]
        n_images = len(case.images)
        exclude = [] if args.no_loo else [cid]
        raw_excl = [c.removeprefix("GT-") for c in exclude]
        registry = base_registry.with_metrics(model_metrics(pilot_records, exclude_cases=raw_excl))
        policy = RoutingPolicy(registry, cfg, use_quality_model=False)
        qm = QualityModel.fit([r for r in rows_for_fit if r.case_id not in exclude], attrs,
                              lam=json.loads((ROOT / "config" / "quality_model.json").read_text())["meta"]["lambda"]
                              if (ROOT / "config" / "quality_model.json").exists() else 10.0, n_boot=20)
        learned = RoutingPolicy(registry, cfg, quality_model=qm, use_quality_model=True)
        fixed_type = heuristic_classify(case.question, image_info=[{}] * n_images).query_type.label

        def pick(p: RoutingPolicy, m: str):
            d = p.route(cls, n_images, m, args.budget, args.live_pool, pool)
            traces.append({"case_id": cid, "mode": m, "learned": p is learned, "classification": cls.to_dict(),
                           "routing": d.to_dict(), "loo_excluded": exclude})
            return d

        choices: dict[str, dict[str, Any]] = {
            "always_cheapest": {"model": policy.cheapest_eligible(n_images, args.live_pool, pool)},
            "always_strongest": {"model": policy.strongest(n_images, args.live_pool, pool)},
            "fixed_task": {"model": policy.fixed_task(fixed_type, n_images, args.live_pool, pool)},
        }
        decisions = {m: pick(policy, m) for m in MODES}
        for m in MODES:
            choices[f"jev_adaptive_{m}"] = {"model": decisions[m].selected_key}
        choices["jev_accuracy_learned"] = {"model": pick(learned, "accuracy").selected_key}

        # Cascade: balanced pick, check its recorded answer, escalate once.
        first = decisions["balanced"].selected_key
        first_risk = heuristic_check(by_case[cid][first].response_text, case.question, cls, case.supplied_evidence).risk
        casc = {"model": first, "escalated": False, "first_model": first, "first_risk": first_risk}
        if first_risk >= thr:
            acc = decisions["accuracy"]
            alt = next((k for k in [acc.selected_key, *acc.fallback_keys] if k and k != first), None)
            if alt:
                alt_risk = heuristic_check(by_case[cid][alt].response_text, case.question, cls, case.supplied_evidence).risk
                casc.update(escalated=True, second_model=alt, second_risk=alt_risk,
                            model=alt if alt_risk <= first_risk else first)
        choices["jev_cascade_balanced"] = casc

        best_score = max(g.score for g in by_case[cid].values())
        for strategy, ch in choices.items():
            key = ch["model"]
            g = by_case[cid].get(key) if key else None
            base = {"case_id": cid, "difficulty": case.difficulty, "strategy": strategy,
                    "jev_query_type": cls.query_type.label, "jev_difficulty": cls.difficulty.label,
                    "classification_source": cls.source}
            if g is None:
                rows.append({**base, "model": None})
                continue
            cost, src = est_cost(g, registry)
            latency = g.latency_seconds
            if ch.get("escalated"):  # pay for both calls
                g1, g2 = by_case[cid][ch["first_model"]], by_case[cid][ch["second_model"]]
                c1, _ = est_cost(g1, registry)
                c2, _ = est_cost(g2, registry)
                cost = (c1 or 0) + (c2 or 0)
                latency = (g1.latency_seconds or 0) + (g2.latency_seconds or 0)
            rows.append({**base, "model": key, "score": g.score, "label": g.label, "unsupported_claim": g.unsupported,
                         "latency_seconds": latency, "cost_usd": cost, "cost_source": src,
                         "score_source": g.score_source, "regret": best_score - g.score,
                         **{k: v for k, v in ch.items() if k != "model"}})
        best = min((g for g in by_case[cid].values() if g.score == best_score),
                   key=lambda g: est_cost(g, registry)[0] or 0)
        cost, src = est_cost(best, registry)
        rows.append({"case_id": cid, "difficulty": case.difficulty, "strategy": "oracle", "model": best.model_key,
                     "score": best.score, "label": best.label, "unsupported_claim": best.unsupported,
                     "latency_seconds": best.latency_seconds, "cost_usd": cost, "cost_source": src,
                     "score_source": best.score_source, "regret": 0.0})

    summary = summarize(rows)
    write_csv(args.output_dir / "offline_eval_rows.csv", rows)
    write_csv(args.output_dir / "offline_eval_summary.csv", summary)
    with (args.output_dir / "offline_routing_traces.jsonl").open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, default=str) + "\n")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classifier": mode, "image_aware": args.image_aware, "leave_one_case_out": not args.no_loo,
        "pool": "currently listed models" if args.live_pool else f"historical pool ({len(pool)} models)",
        "budget_usd": args.budget if args.budget is not None else cfg.get("default_budget_usd_per_query"),
        "n_cases": len(case_ids), "cases": case_ids, "cascade_risk_threshold": thr,
        "score_sources": sorted({g.score_source for g in graded if g.case_id in case_ids}),
    }
    (args.output_dir / "offline_eval_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    cols = ["strategy", "mean_score_0_4", "correct_rate", "unsupported_claim_rate", "avg_cost_usd",
            "avg_latency_seconds", "mean_regret", "escalations", "models_used"]
    print("\n" + " | ".join(cols))
    for s in summary:
        print(" | ".join(str(s[c]) for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
