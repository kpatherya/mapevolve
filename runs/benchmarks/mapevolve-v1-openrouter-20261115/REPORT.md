# MapEvolve benchmark results

Completed: 100 cases × 3 strategies = **300 graded strategy results**.

**These are exploratory automated grades against AI-drafted references.** Human label review and adjudication are still pending. GPT-5.1 graded all answers, including its own; the prior rankings also used GPT-5.1 grades. The text-only grader scores the parsed answer against the frozen reference; structured evidence/confidence fields are retained as diagnostics and are not independently graded.

## Full dataset

| Strategy | Cases | Strict accuracy (4/4) | Score ≥3/4 | Mean score | VLM cost | API time* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cheapest_eligible | 100 | 32.0% | 51.0% | 2.450/4 | $0.0048 | 3.41s |
| jev_adaptive | 100 | 48.0% | 58.0% | 2.760/4 | $0.0102 | 6.06s |
| strongest | 100 | 55.0% | 64.0% | 3.170/4 | $0.2845 | 3.08s |

*API time is the SDK request/response elapsed time, excluding client rate-limit pacing. The original per-answer `latency_seconds` includes client waiting/setup and is retained in the raw results. Neither is a statistically controlled latency benchmark; calls span different periods/provider conditions.

## Landmark identification (50 cases)

| Strategy | Cases | Strict accuracy (4/4) | Score ≥3/4 | Mean score | VLM cost | API time* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cheapest_eligible | 50 | 12.0% | 16.0% | 1.660/4 | $0.0023 | 3.16s |
| jev_adaptive | 50 | 12.0% | 16.0% | 1.680/4 | $0.0047 | 5.63s |
| strongest | 50 | 24.0% | 28.0% | 2.480/4 | $0.1443 | 3.09s |

## Factual retrieval (50 cases)

| Strategy | Cases | Strict accuracy (4/4) | Score ≥3/4 | Mean score | VLM cost | API time* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cheapest_eligible | 50 | 52.0% | 86.0% | 3.240/4 | $0.0025 | 3.67s |
| jev_adaptive | 50 | 84.0% | 100.0% | 3.840/4 | $0.0055 | 6.49s |
| strongest | 50 | 86.0% | 100.0% | 3.860/4 | $0.1402 | 3.08s |

The factual set consists of 30 supplied-building-record questions, 12 supplied-EXIF questions, and 8 image-only unanswerable controls. It measures evidence-grounded response quality, not autonomous web search.

## Earlier-image-overlap not detected (66 cases)

| Strategy | Cases | Strict accuracy (4/4) | Score ≥3/4 | Mean score | VLM cost | API time* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cheapest_eligible | 66 | 31.8% | 45.5% | 2.288/4 | $0.0032 | 3.49s |
| jev_adaptive | 66 | 45.5% | 54.5% | 2.667/4 | $0.0068 | 6.01s |
| strongest | 66 | 56.1% | 62.1% | 3.152/4 | $0.1930 | 3.12s |

The remaining 34 cases reuse images from earlier dataset manifests. Even the 66-case subset shares locations with prior material and is not a location-independent holdout. Multiple questions reuse the same photos; macro-location averages are provided in `location_macro_scores.json`.

## Execution and cost

- Jev classifications: {'jev': 100}. All were reused from saved checkpoints during the funded resumption.
- Answer checks: {'jev_answer_check': 300} strategy rows; shared answers reuse checks.
- Unique exposed calls: classifier: 100, checker: 291, vlm: 291, judge: 291.
- Known unique answer + judge cost: **$0.6499**. TypeSafe dollar costs are not returned by the SDK and are additional/unknown; their token usage is saved.
- Strategy costs attribute the shared answer cost to each strategy that selects it. Do not sum the three strategy totals to calculate actual spend; use `cost_ledger.json`.
- Cheapest = Gemma 3 4B. Strongest = GPT-5.1 by the frozen prior general ranking. Adaptive = balanced routing; escalation, cross-model failover, learned predictor and Pareto pruning disabled.
- After direct OpenAI reported no credits, GPT-5.1 answer/grading calls were explicitly moved to OpenRouter. This backend amendment is saved in `run.json` and `effective_registry.json`; all final strongest answers in this run use that backend.
- Early billing/rate-limit failures were retained. Funded resumption reused successful answers and retried saved API failures; transport changes and runner versions are archived.
- Successful answers were not regenerated or cherry-picked. Same-model strategy selections share a single result. All 100 cases passed the identical-input check; no reference/label fields were found in classifier request states.

## Saved outputs

- `comparison.csv` / `comparison.jsonl`: answers, scores, errors, costs and original client timings.
- `full_diagnostics.csv`: flattened classification probabilities, answerability/tool flags, routing candidates, supplied evidence, answers, checker risks, model confidence, grades and API/client timings.
- `classifications.csv` / `.jsonl`, `routing.jsonl`: every classifier and routing result.
- `summary.csv` / `.json` and `summary_with_api_latency.csv`: breakdowns by type, difficulty, evidence track, location, Jev-only and image-overlap subset.
- `all_sdk_usage.csv`: unique SDK call IDs, model IDs, usage, exposed costs and elapsed times.
- `metadata_crosschecks.csv`, `adjudication_queue.json`: deterministic date-string cross-checks and candidate answers for later review. Literal date matching is diagnostic, not a replacement for grading.
- `cases/`: complete SDK requests/replies, encoded images, raw generated text, parsed JSON, errors (including previous attempts), checks, judge rationale and checkpoints.
- `frozen_inputs/`, `effective_registry.json`, runner copies and `run.json`: configuration and provenance.
- `verification.json`, `output_checksums.json`: completeness, prompt-equality, label-leakage checks and file hashes.

API keys are excluded/redacted. APIs do not expose private internal reasoning or SDK-hidden retry details. No outputs were added to router training or registry ingestion directories.
