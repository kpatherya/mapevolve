# Router tuning on dev (3 cases)

Classifier: heuristic. Registry built from dev grades only, leave-one-case-out.

| Mode | Configs | Default score /4 | Default cost | Tuned score /4 | Tuned cost | Configs tied with best |
| --- | --- | --- | --- | --- | --- | --- |
| economy | 100 | 3.0 | $0.0003255 | 3.0 | $9.63e-05 | 5 |
| balanced | 80 | 3.0 | $0.0003255 | 3.0 | $9.63e-05 | 15 |
| accuracy | 20 | 3.0 | $0.0003255 | 3.0 | $0.0003255 | 20 |

Single-model baselines on the same dev cases:

| Strategy | Score /4 | Cost |
| --- | --- | --- |
| always CLAUDE | 2.0 | $0.010322 |
| always GEMINI | 3.3333 | $0.0254787 |
| always GEMMA | 3.0 | $9.63e-05 |
| always KIMI | 1.0 | $0.0116587 |
| always LLAMA | 2.6667 | $0.0007439 |
| always MISTRAL | 3.3333 | $0.0003582 |
| always OPENAI_FRONTIER_VISION | 3.3333 | $0.002765 |
| always QWEN | 3.0 | $0.0009216 |

Overlay written to `results/tuning/candidate_overlay.yaml` (not applied automatically). It stays out of config/ until dev has >= 20 cases (--force overrides).

**Too few dev cases (3) for these numbers to mean much.** Many configs tie; collect graded data first (scripts.run_live_benchmark + scripts.grade_responses).
