"""Validate all manifests in datasets/ and report coverage gaps.

    python -m scripts.coverage_report [--min-per-cell 5]

Writes results/coverage.md and results/coverage.json. Exits non-zero when a
manifest is invalid (duplicate IDs, missing images, split leakage, ...).
"""

from __future__ import annotations

import argparse
import json
import sys

from jev_router.dataset import ROOT, coverage, coverage_markdown, load_all, validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-per-cell", type=int, default=5)
    args = parser.parse_args()
    cases = load_all()
    errors = validate(cases)
    report = coverage(cases, args.min_per_cell)
    report["validation_errors"] = errors
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "coverage.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = coverage_markdown(report)
    if errors:
        md += "\n## Validation errors\n\n" + "\n".join(f"- {e}" for e in errors) + "\n"
    (out / "coverage.md").write_text(md, encoding="utf-8")
    print(md)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
