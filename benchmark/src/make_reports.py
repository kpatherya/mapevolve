from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from benchmark.src.paths import (
    ARTIFACTS_DIR,
    CASE_IDS,
    FIGURE_DATA_PATH,
    JOINED_RESULTS_PATH,
    LOGICAL_MODEL_ORDER,
    RESOLVED_MODELS_PATH,
)
from benchmark.src.prepare_data import load_cases
from benchmark.src.scoring import failure_interpretation, select_representative_failures

PILOT_SENTENCE = (
    "This six-case pilot suggests that direct VLM comparison is reliable on "
    "some clean aligned cases but becomes less consistent when persistent "
    "changes must be separated from environmental, historical, or subtle visual "
    "differences."
)

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def _float_or_none(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)

def _int_or_none(value: str) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(float(value))

def figure_source() -> dict[str, Any]:
    matrix_rows = _read_csv(ARTIFACTS_DIR / "model_case_matrix.csv")
    difficulty_rows = _read_csv(ARTIFACTS_DIR / "difficulty_summary.csv")
    hallucination_rows = _read_csv(ARTIFACTS_DIR / "hallucination_summary.csv")
    cost_rows = _read_csv(ARTIFACTS_DIR / "cost_latency_summary.csv")
    models = [row["logical_model_name"] for row in matrix_rows]
    scores = []
    for row in matrix_rows:
        scores.append([_int_or_none(row.get(case_id, "")) for case_id in CASE_IDS])
    difficulty = {}
    for row in difficulty_rows:
        if row["logical_model_name"] == "ALL":
            continue
        difficulty.setdefault(row["logical_model_name"], {})[row["difficulty"]] = _float_or_none(row["mean_final_score"])
    rates = {}
    rate_labels = {}
    for row in hallucination_rows:
        if row["difficulty"] != "all":
            continue
        rates[row["logical_model_name"]] = _float_or_none(row["hallucination_rate"])
        rate_labels[row["logical_model_name"]] = f"{row['hallucination_count']}/{row['n_cases']}"
    latency = {}
    for row in cost_rows:
        if row["logical_model_name"] in ("SMOKE_PREFLIGHT", "JUDGE"):
            continue
        latency[row["logical_model_name"]] = _float_or_none(row["mean_latency_seconds"])
    return {
        "models": models,
        "cases": list(CASE_IDS),
        "scores": scores,
        "difficulty_means": difficulty,
        "hallucination_rates": rates,
        "hallucination_labels": rate_labels,
        "latency": latency,
    }

def _plot(source: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = source["models"]
    cases = source["cases"]
    values = np.array(source["scores"], dtype=float)
    masked = np.ma.masked_invalid(values)
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    image = axis.imshow(masked, vmin=0, vmax=4, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(cases)), cases)
    axis.set_yticks(range(len(models)), models)
    axis.set_title("Final score by model and case")
    for row_index, row in enumerate(source["scores"]):
        for col_index, score in enumerate(row):
            label = "—" if score is None else str(score)
            color = "black" if score is None or score >= 3 else "white"
            axis.text(col_index, row_index, label, ha="center", va="center", color=color, fontsize=9)
    figure.colorbar(image, ax=axis, label="Final score (0-4)")
    figure.text(0.01, 0.01, "Six-case pilot. One response per cell. No significance test.", fontsize=8)
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "model_case_score_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    difficulties = ["easy", "medium", "hard"]
    labels = ["Easy", "Medium", "Hard"]
    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(labels))
    width = 0.1
    for index, model in enumerate(models):
        means = [source["difficulty_means"].get(model, {}).get(difficulty) for difficulty in difficulties]
        offset = (index - (len(models) - 1) / 2) * width
        heights = [0 if value is None else value for value in means]
        bars = axis.bar(x + offset, heights, width, label=model)
        for bar, value in zip(bars, means):
            if value is None:
                bar.set_height(0)
                bar.set_alpha(0.15)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 4)
    axis.set_ylabel("Mean final score")
    axis.set_title("Mean final score by difficulty (2 cases each)")
    axis.legend(fontsize=7, ncol=2, frameon=False)
    axis.yaxis.grid(True, linewidth=0.4)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "mean_score_by_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5.2))
    rates = [source["hallucination_rates"].get(model) for model in models]
    heights = [0 if value is None else value for value in rates]
    axis.bar(range(len(models)), heights, color="#4c4c4c")
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Share of scored cases")
    axis.set_title("Unsupported structural-change claims")
    for index, model in enumerate(models):
        axis.text(index, heights[index] + 0.03, source["hallucination_labels"].get(model, ""), ha="center", fontsize=8)
    axis.set_xticks(range(len(models)))
    axis.set_xticklabels(models, rotation=30, ha="right")
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "hallucination_rate_by_model.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5.2))
    latency = [source["latency"].get(model) for model in models]
    heights = [0 if value is None else value for value in latency]
    axis.bar(range(len(models)), heights, color="#4c4c4c")
    axis.set_ylabel("Mean latency (seconds)")
    axis.set_title("Mean response latency")
    axis.set_xticks(range(len(models)))
    axis.set_xticklabels(models, rotation=30, ha="right")
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "latency_by_model.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

def _examples(joined: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in joined:
        if row.get("final_total") is None:
            continue
        if row.get("label") == "correct" and row.get("hallucinated_change") is not True:
            continue
        rows.append(
            {
                "uid": f"{row['logical_model_name']}:{row['case_id']}",
                "case_id": row["case_id"],
                "difficulty": row["difficulty"],
                "model": row["logical_model_name"],
                "resolved_model_id": row.get("resolved_model_id"),
                "final_total": row["final_total"],
                "raw_total": row["raw_total"],
                "label": row["label"],
                "hallucinated_change": row.get("hallucinated_change"),
                "core_change_score": row.get("core_change_score"),
                "confounder_score": row.get("confounder_score"),
                "hallucination_free_score": row.get("hallucination_free_score"),
                "unsupported_points": row.get("unsupported_points") or [],
                "missing_points": row.get("missing_points") or [],
                "judge_reason": row.get("judge_reason") or "",
                "response_text": row.get("response_text") or "",
                "ground_truth": cases[row["case_id"]]["ground_truth_raw"],
            }
        )
    return select_representative_failures(rows, k=4)

def _write_failures(examples: list[dict[str, Any]]) -> None:
    lines = [
        "# Qualitative failures",
        "",
        "Selection rule, applied in order to scored responses that are not labeled correct:",
        "",
        "1. Hallucinated structural-change claims on medium or hard cases, lowest final score first.",
        "2. Remaining hard-case responses, lowest final score first.",
        "3. Remaining medium-case responses, lowest final score first.",
        "4. Prefer a model and case that have not already been selected.",
        "",
        "The examples do not represent all model behavior.",
        "",
    ]
    if not examples:
        lines.append("No scored failure matched this rule.")
    for index, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"## Example {index}",
                "",
                f"- Case: {example['case_id']} ({example['difficulty']})",
                f"- Model: {example['model']} (`{example.get('resolved_model_id')}`)",
                "",
                "Ground truth:",
                "",
                "```text",
                example["ground_truth"].rstrip(),
                "```",
                "",
                "Model response:",
                "",
                "```text",
                (example["response_text"] or "").rstrip(),
                "```",
                "",
                (
                    "- Judge scores: "
                    f"core={example['core_change_score']}, "
                    f"confounder={example['confounder_score']}, "
                    f"hallucination_free={example['hallucination_free_score']}, "
                    f"hallucinated_change={example['hallucinated_change']}, "
                    f"raw_total={example['raw_total']}, "
                    f"final_total={example['final_total']}, "
                    f"label={example['label']}"
                ),
                f"- Unsupported points: {_join(example['unsupported_points'])}",
                f"- Missing points: {_join(example['missing_points'])}",
                f"- Interpretation: {failure_interpretation(example)}",
                "",
            ]
        )
    (ARTIFACTS_DIR / "qualitative_failures.md").write_text("\n".join(lines), encoding="utf-8")

def _join(points: list[str]) -> str:
    if not points:
        return "none recorded"
    return "; ".join(points)

def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines

def _write_digest(joined: list[dict[str, Any]], examples: list[dict[str, Any]]) -> None:
    summary = _read_csv(ARTIFACTS_DIR / "model_summary.csv")
    difficulty = _read_csv(ARTIFACTS_DIR / "difficulty_summary.csv")
    slots = []
    if RESOLVED_MODELS_PATH.exists():
        slots = json.loads(RESOLVED_MODELS_PATH.read_text(encoding="utf-8"))
    successes = [row for row in joined if row.get("status") == "success"]
    failures = [row for row in joined if row.get("status") != "success"]
    scored = [row for row in joined if row.get("final_total") is not None]
    lines = [
        "# MapEvolve VLM pilot digest",
        "",
        "## A. Experimental setup",
        "",
        "- 6 manually curated same-location image pairs.",
        "- 2 easy, 2 medium, and 2 hard cases.",
        "- The correct image pair was supplied directly. The models did not retrieve it.",
        "- 8 VLM slots, one fixed prompt, and one response per model and case.",
        "- No web search, tools, or retrieval.",
        "- A separate text-only LLM judge scored responses against the human answer key.",
        "- `benchmark/artifacts/manual_review.csv` leaves room for a human audit of every output.",
        "",
        "Resolved models:",
        "",
    ]
    if slots:
        lines.extend(
            _md_table(
                ["Slot", "Resolved ID", "Backend", "Status"],
                [
                    [
                        slot.get("logical_name", ""),
                        f"`{slot.get('resolved_model_id') or ''}`",
                        slot.get("backend", ""),
                        slot.get("status", ""),
                    ]
                    for slot in slots
                ],
            )
        )
        substitutions = [slot for slot in slots if slot.get("substitution_reason")]
        if substitutions:
            lines.extend(["", "Substitutions and availability notes:"])
            for slot in substitutions:
                lines.append(f"- {slot.get('logical_name')}: {slot.get('substitution_reason')}")
    lines.extend(
        [
            "",
            f"Successful model responses: {len(successes)} of 48. Scored judgments: {len(scored)}.",
            "",
        ]
    )
    if failures:
        lines.append("Responses that did not succeed:")
        for row in failures:
            lines.append(f"- {row.get('logical_model_name')} {row.get('case_id')}: {row.get('error')}")
        lines.append("")
        lines.append("Missing responses are not interpolated.")
        lines.append("")
    self_judged = [row for row in scored if row.get("self_judged") is True]
    if self_judged:
        judge_model = self_judged[0].get("judge_model")
        lines.append(
            f"{len(self_judged)} scored rows used the same OpenAI model as the judge (`{judge_model}`). "
            "Those rows are marked self_judged and should be included in manual review."
        )
        lines.append("")

    lines.extend(["## B. Main result table", ""])
    lines.extend(
        _md_table(
            ["Model", "Resolved ID", "N", "Mean", "Correct", "Partial", "Incorrect", "Hallucinations", "Mean latency (s)", "Reported cost (USD)"],
            [
                [
                    row["logical_model_name"],
                    f"`{row['resolved_model_id']}`",
                    row["n_cases"],
                    row["mean_final_score"],
                    row["correct_count"],
                    row["partial_count"],
                    row["incorrect_count"],
                    f"{row['hallucination_count']} ({row['hallucination_rate']})",
                    row["mean_latency_seconds"],
                    row["total_reported_cost_usd"] or "not reported",
                ]
                for row in summary
            ],
        )
    )
    lines.extend(
        [
            "",
            "N is the number of judged cases for that model. With two cases per difficulty, percentages are not treated as stable rates.",
            "",
            "## C. Difficulty trend",
            "",
        ]
    )
    all_rows = [row for row in difficulty if row["logical_model_name"] == "ALL"]
    lines.extend(
        _md_table(
            ["Difficulty", "Judged responses", "Mean", "Correct", "Partial", "Incorrect", "Hallucinations"],
            [
                [
                    row["difficulty"],
                    row["n_cases"],
                    row["mean_final_score"],
                    row["correct_count"],
                    row["partial_count"],
                    row["incorrect_count"],
                    row["hallucination_count"],
                ]
                for row in all_rows
            ],
        )
    )
    lines.extend(["", "Per model, means and counts use the two cases in that difficulty:", ""])
    per_model = [row for row in difficulty if row["logical_model_name"] != "ALL"]
    compact: dict[str, dict[str, str]] = {}
    for row in per_model:
        compact.setdefault(row["logical_model_name"], {})[row["difficulty"]] = (
            f"{row['mean_final_score'] or '—'} "
            f"({row['correct_count']}/{row['partial_count']}/{row['incorrect_count']}, h={row['hallucination_count']})"
        )
    lines.extend(
        _md_table(
            ["Model", "Easy", "Medium", "Hard"],
            [
                [
                    name,
                    compact.get(name, {}).get("easy", "—"),
                    compact.get(name, {}).get("medium", "—"),
                    compact.get(name, {}).get("hard", "—"),
                ]
                for name in LOGICAL_MODEL_ORDER
                if name in compact
            ],
        )
    )
    lines.extend(
        [
            "",
            "Each cell is mean final score, then correct/partial/incorrect counts, then the hallucination count. Each difficulty has two cases.",
            "",
            "## D. Hallucination / unsupported-change behavior",
            "",
            "A hallucinated change is a concrete physical or structural change that the answer key does not support. The final score of those responses is capped at 3.",
            "",
        ]
    )
    total_hallucinations = sum(int(row["hallucination_count"]) for row in summary)
    total_scored = sum(int(row["n_cases"]) for row in summary)
    lines.append(
        f"The judge marked {total_hallucinations} of {total_scored} scored responses as containing an unsupported structural-change claim."
    )
    lines.append("")
    lines.extend(
        _md_table(
            ["Model", "Hallucination count", "Scored cases"],
            [[row["logical_model_name"], row["hallucination_count"], row["n_cases"]] for row in summary],
        )
    )
    lines.extend(["", "## E. Representative failure examples", ""])
    if not examples:
        lines.append("No scored failure was available for the selection rule.")
    for index, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"### {index}. {example['model']} on {example['case_id']}",
                "",
                f"Final score {example['final_total']} ({example['label']}); hallucinated_change={example['hallucinated_change']}.",
                "",
                failure_interpretation(example),
                "",
                "Response:",
                "",
                "```text",
                (example["response_text"] or "").rstrip(),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "These examples were selected by the rule in `qualitative_failures.md`. They do not represent all behavior.",
            "",
            "## F. Interpretation",
            "",
            PILOT_SENTENCE,
            "",
        ]
    )
    if all_rows:
        bits = []
        for row in all_rows:
            bits.append(
                f"{row['difficulty']}: mean {row['mean_final_score'] or 'n/a'} "
                f"on {row['n_cases']} judgments "
                f"({row['correct_count']} correct, {row['partial_count']} partial, {row['incorrect_count']} incorrect, "
                f"{row['hallucination_count']} with an unsupported structural claim)"
            )
        lines.append("Pooled counts — " + "; ".join(bits) + ".")
        lines.append("")
    lines.extend(
        [
            "The table is a description of this pilot, not a ranking, and it does not estimate performance on other locations or time gaps.",
            "",
            "## G. Motivation for MapEvolve",
            "",
            "The benchmark gives every VLM the correct pair of images manually.",
            "",
            "Therefore failures are downstream visual reasoning failures, not retrieval failures.",
            "",
            "A real MapEvolve query is harder because the system must first retrieve and verify the correct historical evidence.",
            "",
            "This motivates a pipeline that separates retrieval, pair verification, evidence tracing or refusal, and final VLM generation.",
            "",
            "Figures:",
            "",
            "- `benchmark/artifacts/model_case_score_matrix.png`",
            "- `benchmark/artifacts/mean_score_by_difficulty.png`",
            "- `benchmark/artifacts/hallucination_rate_by_model.png`",
            "- `benchmark/artifacts/latency_by_model.png`",
            "",
        ]
    )
    (ARTIFACTS_DIR / "presentation_digest.md").write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    if not JOINED_RESULTS_PATH.exists():
        raise SystemExit("joined_results.json is missing. Run python -m benchmark.src.aggregate_results first.")
    joined = json.loads(JOINED_RESULTS_PATH.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in load_cases()}
    source = figure_source()
    FIGURE_DATA_PATH.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    _plot(source)
    examples = _examples(joined, cases)
    _write_failures(examples)
    _write_digest(joined, examples)
    print("Wrote figures, qualitative_failures.md, and presentation_digest.md")

if __name__ == "__main__":
    main()
