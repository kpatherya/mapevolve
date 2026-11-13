"""Tune routing settings on the dev split; evaluate on test once.

    python -m scripts.tune_router                       # grid search on dev, writes config/routing.tuned.yaml
    python -m scripts.tune_router --evaluate-test       # score the tuned overlay on test (do this once)

Rules this script enforces:
  * Tuning sees only dev cases, and the registry used while tuning is built
    from dev grades only (leave-one-case-out inside dev).
  * --evaluate-test builds the registry from all dev grades, routes the test
    cases with the frozen overlay, and appends to results/tuning/test_runs.jsonl.
    It warns if test was already scored with a different overlay.

Grid (per mode, since each mode reads different settings):
  economy   min_quality x shrinkage_k x unsupported_penalty
  balanced  weights x shrinkage_k x unsupported_penalty
  accuracy  shrinkage_k x unsupported_penalty

Selection per mode:
  economy   cheapest config whose mean score is within --tolerance of the best economy config
  balanced  highest mean score, ties -> lower cost
  accuracy  highest mean score, ties -> fewer unsupported claims, then lower cost
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import yaml

from jev_router.offline import deep_merge, fixed_choice, load_offline, route_all, score_choices
from jev_router.registry import ROOT, load_routing_config

OUT = ROOT / "results" / "tuning"
TUNED = ROOT / "config" / "routing.tuned.yaml"

K = [0.5, 1, 2, 4, 8]
PENALTY = [0.0, 0.1, 0.25, 0.5]
FLOOR = [0.6, 0.65, 0.7, 0.75, 0.8]
WEIGHTS = [(0.4, 0.36, 0.24), (0.6, 0.24, 0.16), (0.8, 0.12, 0.08), (0.9, 0.06, 0.04)]


def grid(mode: str) -> list[dict[str, Any]]:
    out = []
    if mode == "economy":
        for f, k, pen in itertools.product(FLOOR, K, PENALTY):
            out.append({"registry": {"shrinkage_k": k}, "modes": {"economy": {"min_quality": f, "unsupported_penalty": pen}}})
    elif mode == "balanced":
        for w, k, pen in itertools.product(WEIGHTS, K, PENALTY):
            out.append({"registry": {"shrinkage_k": k}, "modes": {"balanced": {
                "weights": {"quality": w[0], "cost": w[1], "latency": w[2]}, "unsupported_penalty": pen}}})
    else:
        for k, pen in itertools.product(K, PENALTY):
            out.append({"registry": {"shrinkage_k": k}, "modes": {"accuracy": {"unsupported_penalty": pen}}})
    return out


def flat(overlay: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in overlay.items():
        key = f"{prefix}{k}"
        out.update(flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def changes(row: dict[str, Any], base: dict[str, Any]) -> int:
    """How many tuned settings differ from the current config (ties prefer fewer)."""
    n = 0
    for k, v in row.items():
        if not (k.startswith("modes.") or k.startswith("registry.")):
            continue
        cur: Any = base
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        n += cur != v
    return n


def pick(mode: str, rows: list[dict[str, Any]], tolerance: float, base: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in rows if r.get("n_cases")]
    cost = lambda r: r["avg_cost_usd"] if r["avg_cost_usd"] is not None else 1e9
    if mode == "economy":
        best = max(r["mean_score_0_4"] for r in rows)
        ok = [r for r in rows if r["mean_score_0_4"] >= best - tolerance]
        return min(ok, key=lambda r: (round(cost(r), 9), -r["mean_score_0_4"], changes(r, base)))
    if mode == "balanced":
        return min(rows, key=lambda r: (-r["mean_score_0_4"], round(cost(r), 9), changes(r, base)))
    return min(rows, key=lambda r: (-r["mean_score_0_4"], r["unsupported_rate"], round(cost(r), 9), changes(r, base)))


def overlay_hash(overlay: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(overlay, sort_keys=True).encode()).hexdigest()[:12]


def tune(args, classifier: str) -> int:
    base = load_routing_config()
    data = load_offline(classifier, args.image_aware, split="dev", cfg=base, registry_split="dev")
    if len(data.case_ids) < 2:
        print("error: need at least 2 dev cases graded for every model", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    chosen: dict[str, Any] = {}
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "classifier": classifier,
               "dev_cases": data.case_ids, "pool": data.pool, "modes": {}}
    baselines = {f"always {k}": score_choices(data, fixed_choice(data, k)) for k in data.pool}
    for mode in ("economy", "balanced", "accuracy"):
        rows = []
        default = score_choices(data, route_all(data, base, mode, args.budget))
        for ov in grid(mode):
            cfg = deep_merge(base, ov)
            rows.append({**flat(ov), **score_choices(data, route_all(data, cfg, mode, args.budget))})
        with (OUT / f"grid_{mode}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        best = pick(mode, rows, args.tolerance, base)
        n_tied = sum(1 for r in rows if (r["mean_score_0_4"], r["avg_cost_usd"]) == (best["mean_score_0_4"], best["avg_cost_usd"]))
        summary["modes"][mode] = {"default": default, "best": best, "n_configs": len(rows), "n_configs_tied_with_best": n_tied}
        # Keep the mode-specific keys; shrinkage_k is shared, taken from accuracy last.
        chosen = deep_merge(chosen, {"modes": {mode: {k.split(".", 2)[2]: v for k, v in best.items()
                                                       if k.startswith(f"modes.{mode}.")}}})
        if "registry.shrinkage_k" in best:
            chosen = deep_merge(chosen, {"registry": {"shrinkage_k": best["registry.shrinkage_k"]}})
    # Nest dotted weight keys back into dicts.
    for mode, settings in chosen.get("modes", {}).items():
        nested: dict[str, Any] = {}
        for k, v in settings.items():
            parts = k.split(".")
            cur = nested
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = v
        chosen["modes"][mode] = nested
    header = (f"# Tuned on dev ({len(data.case_ids)} cases: {', '.join(data.case_ids)}) by scripts.tune_router\n"
              f"# {summary['generated_at']}, classifier={classifier}. Apply with JEV_ROUTING_OVERLAY={TUNED.relative_to(ROOT)}\n"
              f"# Overlay hash {overlay_hash(chosen)}. Score it on test once: python -m scripts.tune_router --evaluate-test\n")
    enough = len(data.case_ids) >= args.min_dev_cases or args.force
    target = TUNED if enough else OUT / "candidate_overlay.yaml"
    target.write_text(header + yaml.safe_dump(chosen, sort_keys=True), encoding="utf-8")
    summary["overlay"] = chosen
    summary["baselines"] = baselines
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [f"# Router tuning on dev ({len(data.case_ids)} cases)", "",
             f"Classifier: {classifier}. Registry built from dev grades only, leave-one-case-out.", "",
             "| Mode | Configs | Default score /4 | Default cost | Tuned score /4 | Tuned cost | Configs tied with best |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for mode, m in summary["modes"].items():
        d, b = m["default"], m["best"]
        lines.append(f"| {mode} | {m['n_configs']} | {d['mean_score_0_4']} | ${d['avg_cost_usd']} | "
                     f"{b['mean_score_0_4']} | ${b['avg_cost_usd']} | {m['n_configs_tied_with_best']} |")
    lines += ["", "Single-model baselines on the same dev cases:", "", "| Strategy | Score /4 | Cost |", "| --- | --- | --- |"]
    lines += [f"| {k} | {v['mean_score_0_4']} | ${v['avg_cost_usd']} |" for k, v in baselines.items()]
    lines += ["", f"Overlay written to `{target.relative_to(ROOT)}` (not applied automatically)."
              + ("" if enough else f" It stays out of config/ until dev has >= {args.min_dev_cases} cases (--force overrides)."), ""]
    if len(data.case_ids) < 20:
        lines += [f"**Too few dev cases ({len(data.case_ids)}) for these numbers to mean much.** Many configs tie; "
                  "collect graded data first (scripts.run_live_benchmark + scripts.grade_responses).", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


def evaluate_test(args, classifier: str) -> int:
    if not TUNED.exists():
        print("error: run tuning first (config/routing.tuned.yaml missing)", file=sys.stderr)
        return 2
    overlay = yaml.safe_load(TUNED.read_text(encoding="utf-8")) or {}
    h = overlay_hash(overlay)
    log = OUT / "test_runs.jsonl"
    previous = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    if any(p["overlay_hash"] != h for p in previous):
        print("WARNING: test was already scored with a different overlay. Repeated test looks turn test into dev; "
              "report the first result or collect fresh test cases.", file=sys.stderr)
    cfg = load_routing_config(overlay=TUNED)
    data = load_offline(classifier, args.image_aware, split="test", cfg=cfg, registry_split="dev")
    if not data.case_ids:
        print("error: no test cases graded for every model", file=sys.stderr)
        return 2
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "overlay_hash": h,
              "classifier": classifier, "test_cases": data.case_ids,
              "modes": {m: score_choices(data, route_all(data, cfg, m, args.budget, loo=False))
                        for m in ("economy", "balanced", "accuracy")},
              "baselines": {f"always {k}": score_choices(data, fixed_choice(data, k)) for k in data.pool}}
    OUT.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", choices=("auto", "jev", "heuristic"), default="auto")
    parser.add_argument("--image-aware", action="store_true")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=0.25, help="Economy: score slack (0-4 scale) for the cheapest pick")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--min-dev-cases", type=int, default=20,
                        help="Below this, the overlay goes to results/tuning/ instead of config/")
    parser.add_argument("--force", action="store_true", help="Write config/routing.tuned.yaml even with few dev cases")
    args = parser.parse_args()
    classifier = args.classifier if args.classifier != "auto" else ("jev" if os.environ.get("TYPESAFE_API_KEY") else "heuristic")
    return evaluate_test(args, classifier) if args.evaluate_test else tune(args, classifier)


if __name__ == "__main__":
    sys.exit(main())
