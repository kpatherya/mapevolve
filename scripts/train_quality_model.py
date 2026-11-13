"""Train the learned quality predictor and compare it with the registry table.

    python -m scripts.train_quality_model [--classifier heuristic|jev] [--image-aware]

Evaluation is leave-one-case-out (LOCO): for each graded case, every
estimator is refit without that case and asked to predict all models' scores
on it. Reported per estimator:
  * MAE of the predicted normalized score (score / 4)
  * top-1 routing: realized score of the model each estimator ranks first,
    and regret versus the best model on that case

Estimators: general mean per model, the registry's shrunk task x difficulty
table (current policy), and the learned predictor. Ridge strength is chosen
on the same LOCO folds (slightly optimistic; noted in the output).

Writes config/quality_model.json and results/quality_model_eval.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from statistics import mean

from dotenv import load_dotenv

from jev_router.benchmark_data import load_responses, model_metrics
from jev_router.classify_cache import classify_cases, resolve_image_aware
from jev_router.dataset import ROOT, load_all
from jev_router.graded import load_graded
from jev_router.quality_model import QualityModel, TrainingRow, model_attributes
from jev_router.registry import ModelRegistry, load_routing_config

OUT_MODEL = ROOT / "config" / "quality_model.json"


def registry_estimate(registry: ModelRegistry, cls, key: str) -> float:
    q = 0.0
    for t, pt in cls.query_type.probabilities.items():
        for d, pd in cls.difficulty.probabilities.items():
            est = registry.score(key, t, d)
            q += pt * pd * (est.quality if est.quality is not None else 0.5)
    norm = sum(cls.query_type.probabilities.values()) * sum(cls.difficulty.probabilities.values())
    return q / norm if norm else 0.5


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", choices=("auto", "jev", "heuristic"), default="auto")
    parser.add_argument("--image-aware", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--n-boot", type=int, default=40)
    args = parser.parse_args()
    mode = args.classifier if args.classifier != "auto" else ("jev" if os.environ.get("TYPESAFE_API_KEY") else "heuristic")

    cfg = load_routing_config()
    base_registry = ModelRegistry.load(routing=cfg)
    attrs = model_attributes(base_registry.data["models"])
    graded = load_graded()
    cases = {c.case_id: c for c in load_all()}
    graded = [g for g in graded if g.case_id in cases]
    args.image_aware = resolve_image_aware(args.image_aware, mode)
    cls = classify_cases([cases[cid] for cid in sorted({g.case_id for g in graded})], mode, args.image_aware)
    rows = [TrainingRow(g.case_id, g.model_key, g.score, g.unsupported, cls[g.case_id]) for g in graded]
    case_ids = sorted({r.case_id for r in rows})
    by_case = defaultdict(dict)
    for r in rows:
        by_case[r.case_id][r.model_key] = r
    pilot_records = load_responses()

    def loco(estimator):
        preds = {}
        for cid in case_ids:
            train = [r for r in rows if r.case_id != cid]
            preds[cid] = estimator(train, cid)
        return preds

    def general_est(train, cid):
        m = defaultdict(list)
        for r in train:
            m[r.model_key].append(r.score / 4)
        return {k: mean(v) for k, v in m.items()}

    def registry_est(train, cid):
        raw = cid.removeprefix("GT-")
        reg = base_registry.with_metrics(model_metrics(pilot_records, exclude_cases=[raw]))
        return {k: registry_estimate(reg, cls[cid], k) for k in by_case[cid]}

    def learned_est_factory(lam):
        def est(train, cid):
            qm = QualityModel.fit(train, attrs, lam=lam, n_boot=1)
            return {k: qm.predict(cls[cid], k).quality for k in by_case[cid]}
        return est

    def score(preds):
        errs, realized, regrets = [], [], []
        for cid, p in preds.items():
            truth = {k: r.score / 4 for k, r in by_case[cid].items()}
            errs += [abs(p[k] - truth[k]) for k in truth if k in p]
            pick = max(p, key=lambda k: (round(p[k], 6), -attrs[k]["log_price_in"]))
            realized.append(truth[pick] * 4)
            regrets.append(max(truth.values()) * 4 - truth[pick] * 4)
        return {"mae_normalized": round(mean(errs), 4), "top1_mean_score_0_4": round(mean(realized), 4),
                "top1_mean_regret": round(mean(regrets), 4)}

    results = {"general_mean": score(loco(general_est)), "registry_shrunk_table": score(loco(registry_est))}
    grid = [0.3, 1.0, 3.0, 10.0, 30.0]
    lam_scores = {lam: score(loco(learned_est_factory(lam))) for lam in grid}
    best_lam = min(grid, key=lambda lam: lam_scores[lam]["mae_normalized"])
    results["learned_predictor"] = {**lam_scores[best_lam], "lambda": best_lam}
    results["learned_predictor_by_lambda"] = {str(k): v for k, v in lam_scores.items()}

    final = QualityModel.fit(rows, attrs, lam=best_lam, n_boot=args.n_boot)
    final.meta.update({"classifier": mode, "image_aware": args.image_aware,
                       "query_types": sorted({cases[c].query_type for c in case_ids}),
                       "score_sources": sorted({g.score_source for g in graded})})
    final.save(OUT_MODEL)

    report = {"classifier": mode, "n_rows": len(rows), "n_cases": len(case_ids),
              "trained_query_types": final.meta["query_types"], "results": results,
              "note": "Lambda chosen on the same LOCO folds; the learned MAE is slightly optimistic."}
    (ROOT / "results" / "quality_model_eval.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Quality predictor — LOCO evaluation ({len(rows)} graded responses, {len(case_ids)} cases)", "",
             f"Classifier features: {mode}. Trained query types: {final.meta['query_types']}.", "",
             "| Estimator | MAE (score/4) | Top-1 mean score /4 | Top-1 regret |", "| --- | --- | --- | --- |"]
    for name in ("general_mean", "registry_shrunk_table", "learned_predictor"):
        r = results[name]
        lines.append(f"| {name}{' (λ=' + str(r['lambda']) + ')' if 'lambda' in r else ''} | {r['mae_normalized']} | "
                     f"{r['top1_mean_score_0_4']} | {r['top1_mean_regret']} |")
    lines += ["", report["note"], ""]
    (ROOT / "results" / "quality_model_eval.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved {OUT_MODEL.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
