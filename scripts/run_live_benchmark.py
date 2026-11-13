"""Run manifest cases through VLMs to collect responses for grading.

    python -m scripts.run_live_benchmark --dataset demo_world --dry-run
    python -m scripts.run_live_benchmark --dataset demo_world --models GEMMA MISTRAL --max-cost 0.50

Each case is sent exactly as the router would send it: same system prompt,
the task block for Jev's query type, supplied evidence, and the action note
(pair check / coordinate-tool result). Declined cases are still run so their
refusal behaviour can be graded.

Responses go to results/live_runs/<run_id>.jsonl. Re-running with the same
--run-id skips (case, model) pairs already done. --dry-run estimates cost
from the registry without calling any API. Needs OPENROUTER_API_KEY and/or
OPENAI_API_KEY (and TYPESAFE_API_KEY for Jev classification).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

from jev_router.actions import decide_action
from jev_router.classify_cache import classify_cases
from jev_router.dataset import ROOT, load_all
from jev_router.registry import ModelRegistry, load_routing_config
from jev_router.vlm import VLMExecutor

LIVE_DIR = ROOT / "results" / "live_runs"


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None, help="Registry keys (default: all listed models)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-cost", type=float, default=1.0, help="Stop once estimated spend exceeds this (USD)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Run each (case, model) this many times (run IDs <run>-r1, -r2, ...) for consistency")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_routing_config()
    registry = ModelRegistry.load(routing=cfg)
    cases = [c for c in load_all() if (args.dataset is None or c.dataset == args.dataset)
             and (args.split == "all" or c.split == args.split)
             and (args.cases is None or c.case_id in args.cases)]
    models = args.models or [k for k in registry.keys() if registry.is_listed(k)]
    missing = [m for m in models if m not in registry.models]
    if missing:
        print(f"error: unknown model keys {missing}", file=sys.stderr)
        return 2
    pairs = [(c, m) for c in cases for m in models if registry.supports_images(m, len(c.images))]
    plan = [(c, m, k) for k in range(1, max(1, args.repeats) + 1) for c, m in pairs]
    est = sum((registry.estimate_cost(m, len(c.images))[0] or 0) for c, m, _ in plan)
    print(f"{len(cases)} cases x {len(models)} models x {max(1, args.repeats)} repeat(s) = {len(plan)} calls, "
          f"estimated ${est:.4f} (cap ${args.max_cost:.2f})")
    if args.dry_run:
        return 0
    needed = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}
    for m in models:
        var = needed[registry.get(m)["backend"]]
        if not os.environ.get(var):
            print(f"error: {var} is required for {m}", file=sys.stderr)
            return 2

    mode = "jev" if os.environ.get("TYPESAFE_API_KEY") else "heuristic"
    cls = classify_cases(cases, mode)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("live-%Y%m%dT%H%M%SZ")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = LIVE_DIR / f"{run_id}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["status"] == "ok":
                done.add((r["case_id"], r["model_key"], r["run_id"]))
    executor = VLMExecutor(registry, cfg["generation"])
    spent = 0.0
    with path.open("a", encoding="utf-8") as fh:
        for case, key, k in plan:
            row_run = run_id if args.repeats <= 1 else f"{run_id}-r{k}"
            if (case.case_id, key, row_run) in done:
                continue
            if spent >= args.max_cost:
                print(f"stopping: spend ${spent:.4f} reached --max-cost")
                break
            c = cls[case.case_id]
            action = decide_action(c, cfg, case.supplied_evidence)
            res = executor.run(key, case.question, case.image_paths(), task=c.query_type.label,
                               evidence=case.supplied_evidence, note=action.prompt_note)
            spent += res.cost_usd or (registry.estimate_cost(key, len(case.images))[0] or 0)
            row = {"run_id": row_run, "case_id": case.case_id, "model_key": key, "model_id": res.model_id,
                   "status": res.status, "answer": res.answer, "structured": res.structured, "raw_text": res.raw_text,
                   "latency_seconds": res.latency_seconds, "input_tokens": res.input_tokens,
                   "output_tokens": res.output_tokens, "cost_usd": res.cost_usd, "cost_source": res.cost_source,
                   "error": res.error, "action": action.action, "classification_source": c.source,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            print(f"{res.status:<9} {case.case_id:<26} {key:<24} {res.latency_seconds}s ${res.cost_usd}")
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"wrote {shown} (spent ~${spent:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
