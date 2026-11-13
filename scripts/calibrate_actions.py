"""Choose action thresholds for one classifier on dev cases only.

    python -m scripts.calibrate_actions --classifier jev --image-aware
    python -m scripts.calibrate_actions --classifier jev --image-aware --evaluate-test   # once, after freezing

Jev and the heuristic fallback put their probabilities on different scales, so
one shared 0.5 cut-off suits neither. This searches the per-flag thresholds
(tools, retrieval, pair verification and its signal) for the best action
accuracy on dev and prints the `actions.by_source.<source>` block to put in
config/routing.yaml. Ties prefer the current settings. The decline threshold is
searched only when dev has decline cases; otherwise it stays as configured.

Uses cached classifications (results/classifications/), so it makes no API
calls once evaluate_classifier has run. Writes results/calibration/actions_<tag>.{json,md}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from typing import Any

from dotenv import load_dotenv

from jev_router.actions import action_thresholds, decide_action
from jev_router.classify_cache import classify_cases, resolve_image_aware
from jev_router.dataset import ACTIONS, ROOT, load_all
from jev_router.offline import deep_merge
from jev_router.registry import load_routing_config

OUT = ROOT / "results" / "calibration"
GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
DECLINE_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
KEYS = ("tools_threshold", "retrieval_threshold", "verify_signal", "verify_threshold", "decline_if_answerable_below")


def accuracy(cases, cls, cfg) -> tuple[float, Counter]:
    conf: Counter = Counter()
    for c in cases:
        conf[(c.expected_action, decide_action(cls[c.case_id], cfg, c.supplied_evidence).action)] += 1
    return (sum(v for (e, a), v in conf.items() if e == a) / len(cases) if cases else 0.0), conf


def with_thresholds(cfg: dict[str, Any], source: str, th: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(cfg, {"actions": {"by_source": {source: th}}})


def distance(th: dict[str, Any], base: dict[str, Any]) -> float:
    return sum(abs(th[k] - base[k]) if isinstance(th[k], float) else float(th[k] != base[k]) for k in KEYS)


def confusion_md(conf: Counter) -> list[str]:
    lines = ["| expected \\ chosen | " + " | ".join(ACTIONS) + " |", "| --- |" + " --- |" * len(ACTIONS)]
    for e in ACTIONS:
        lines.append(f"| {e} | " + " | ".join(str(conf.get((e, a), 0)) for a in ACTIONS) + " |")
    return lines


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", choices=("jev", "heuristic"), required=True)
    parser.add_argument("--image-aware", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--evaluate-test", action="store_true", help="Also score the chosen thresholds on test")
    args = parser.parse_args()

    cfg = load_routing_config()
    source = "jev" if args.classifier == "jev" else "heuristic_fallback"
    cases = load_all()
    args.image_aware = resolve_image_aware(args.image_aware, args.classifier)
    tag = f"{args.classifier}{'_img' if args.image_aware else ''}"
    cls = classify_cases(cases, args.classifier, args.image_aware)
    fell_back = sum(1 for c in cls.values() if c.source != source)
    if fell_back:
        print(f"error: {fell_back}/{len(cls)} classifications are not from {source}; rerun evaluate_classifier "
              f"with a working key first", file=sys.stderr)
        return 2
    dev = [c for c in cases if c.split == "dev"]
    test = [c for c in cases if c.split == "test"]
    base = {k: action_thresholds(cfg, source)[k] for k in KEYS}
    dev_has_decline = any(c.expected_action == "decline" for c in dev)

    best, best_key = None, None
    for tools, retr, signal, verify, decline in itertools.product(
            GRID, GRID, ("flag", "query_type"), GRID, DECLINE_GRID if dev_has_decline else [base[KEYS[4]]]):
        th = dict(zip(KEYS, (tools, retr, signal, verify, decline)))
        acc, _ = accuracy(dev, cls, with_thresholds(cfg, source, th))
        key = (acc, -distance(th, base))
        if best_key is None or key > best_key:
            best, best_key = th, key

    base_dev, base_conf = accuracy(dev, cls, cfg)
    new_cfg = with_thresholds(cfg, source, best)
    new_dev, new_conf = accuracy(dev, cls, new_cfg)
    report: dict[str, Any] = {
        "classifier": tag, "source": source, "n_dev": len(dev), "dev_expected": dict(Counter(c.expected_action for c in dev)),
        "current": base, "chosen": best, "dev_accuracy_current": round(base_dev, 4), "dev_accuracy_chosen": round(new_dev, 4),
        "decline_calibrated": dev_has_decline,
    }
    lines = [f"# Action threshold calibration — {tag} (dev only, {len(dev)} cases)", "",
             f"Dev action accuracy: {base_dev:.3f} with the current thresholds -> **{new_dev:.3f}** with the chosen ones.", "",
             "| Setting | Current | Chosen |", "| --- | --- | --- |"]
    lines += [f"| {k} | {base[k]} | {best[k]} |" for k in KEYS]
    if not dev_has_decline:
        lines += ["", "Dev has no decline cases, so the decline threshold was not searched."]
    lines += ["", "## Dev confusion, chosen thresholds", ""] + confusion_md(new_conf)
    if args.evaluate_test:
        base_test, _ = accuracy(test, cls, cfg)
        new_test, test_conf = accuracy(test, cls, new_cfg)
        report.update(n_test=len(test), test_accuracy_current=round(base_test, 4), test_accuracy_chosen=round(new_test, 4))
        lines += ["", f"## Test ({len(test)} cases, scored once)", "",
                  f"Test action accuracy: {base_test:.3f} current -> **{new_test:.3f}** chosen.", ""] + confusion_md(test_conf)
    snippet = "actions:\n  by_source:\n    " + source + ":\n" + "".join(
        f"      {k}: {best[k]}\n" for k in KEYS)
    lines += ["", "## Config block", "", "```yaml", snippet.rstrip(), "```"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"actions_{tag}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUT / f"actions_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
