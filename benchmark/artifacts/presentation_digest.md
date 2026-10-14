# MapEvolve VLM pilot digest

## A. Experimental setup

- 6 manually curated same-location image pairs.
- 2 easy, 2 medium, and 2 hard cases.
- The correct image pair was supplied directly. The models did not retrieve it.
- 8 VLM slots, one fixed prompt, and one response per model and case.
- No web search, tools, or retrieval.
- A separate text-only LLM judge scored responses against the human answer key.
- `benchmark/artifacts/manual_review.csv` leaves room for a human audit of every output.

Resolved models:

| Slot | Resolved ID | Backend | Status |
| --- | --- | --- | --- |
| OPENAI_FRONTIER_VISION | `gpt-5.1-2025-11-13` | openai | ready |
| GEMINI | `google/gemini-3.1-pro-preview` | openrouter | ready |
| CLAUDE | `anthropic/claude-sonnet-4` | openrouter | ready |
| QWEN | `qwen/qwen3-vl-235b-a22b-instruct` | openrouter | ready |
| GEMMA | `google/gemma-3-27b-it` | openrouter | ready |
| LLAMA | `meta-llama/llama-4-maverick` | openrouter | ready |
| MISTRAL | `mistralai/mistral-small-3.2-24b-instruct` | openrouter | ready |
| KIMI | `moonshotai/kimi-k2.5` | openrouter | substituted |

Substitutions and availability notes:
- OPENAI_FRONTIER_VISION: API returned snapshot id gpt-5.1-2025-11-13 for requested gpt-5.1.
- KIMI: exact id moonshotai/kimi-vl-a3b-thinking is unavailable or not image-capable; using closest current image-capable KIMI family model moonshotai/kimi-k2.5

Successful model responses: 48 of 48. Scored judgments: 48.

6 scored rows used the same OpenAI model as the judge (`gpt-5.1-2025-11-13`). Those rows are marked self_judged and should be included in manual review.

## B. Main result table

| Model | Resolved ID | N | Mean | Correct | Partial | Incorrect | Hallucinations | Mean latency (s) | Reported cost (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OPENAI_FRONTIER_VISION | `gpt-5.1-2025-11-13` | 6 | 3.0000 | 3 | 3 | 0 | 3 (0.5000) | 12.2858 | not reported |
| GEMINI | `google/gemini-3.1-pro-preview` | 6 | 2.8333 | 3 | 2 | 1 | 3 (0.5000) | 19.6518 | 0.1520 |
| CLAUDE | `anthropic/claude-sonnet-4` | 6 | 2.6667 | 2 | 3 | 1 | 4 (0.6667) | 5.8939 | 0.0637 |
| QWEN | `qwen/qwen3-vl-235b-a22b-instruct` | 6 | 3.0000 | 4 | 0 | 2 | 2 (0.3333) | 16.7236 | 0.0059 |
| GEMMA | `google/gemma-3-27b-it` | 6 | 3.0000 | 3 | 2 | 1 | 1 (0.1667) | 4.9317 | 0.0006 |
| LLAMA | `meta-llama/llama-4-maverick` | 6 | 2.6667 | 3 | 2 | 1 | 0 (0.0000) | 7.6181 | 0.0047 |
| MISTRAL | `mistralai/mistral-small-3.2-24b-instruct` | 6 | 3.1667 | 4 | 1 | 1 | 1 (0.1667) | 7.9817 | 0.0022 |
| KIMI | `moonshotai/kimi-k2.5` | 6 | 1.8333 | 1 | 3 | 2 | 5 (0.8333) | 74.2626 | 0.0695 |

N is the number of judged cases for that model. With two cases per difficulty, percentages are not treated as stable rates.

## C. Difficulty trend

| Difficulty | Judged responses | Mean | Correct | Partial | Incorrect | Hallucinations |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 16 | 3.6250 | 14 | 1 | 1 | 2 |
| medium | 16 | 3.2500 | 9 | 6 | 1 | 5 |
| hard | 16 | 1.4375 | 0 | 9 | 7 | 12 |

Per model, means and counts use the two cases in that difficulty:

| Model | Easy | Medium | Hard |
| --- | --- | --- | --- |
| OPENAI_FRONTIER_VISION | 4.0000 (2/0/0, h=0) | 3.0000 (1/1/0, h=1) | 2.0000 (0/2/0, h=2) |
| GEMINI | 4.0000 (2/0/0, h=0) | 2.5000 (1/0/1, h=1) | 2.0000 (0/2/0, h=2) |
| CLAUDE | 3.0000 (1/1/0, h=1) | 3.5000 (1/1/0, h=1) | 1.5000 (0/1/1, h=2) |
| QWEN | 4.0000 (2/0/0, h=0) | 4.0000 (2/0/0, h=0) | 1.0000 (0/0/2, h=2) |
| GEMMA | 4.0000 (2/0/0, h=0) | 3.5000 (1/1/0, h=0) | 1.5000 (0/1/1, h=1) |
| LLAMA | 4.0000 (2/0/0, h=0) | 3.0000 (1/1/0, h=0) | 1.0000 (0/1/1, h=0) |
| MISTRAL | 4.0000 (2/0/0, h=0) | 4.0000 (2/0/0, h=0) | 1.5000 (0/1/1, h=1) |
| KIMI | 2.0000 (1/0/1, h=1) | 2.5000 (0/2/0, h=2) | 1.0000 (0/1/1, h=2) |

Each cell is mean final score, then correct/partial/incorrect counts, then the hallucination count. Each difficulty has two cases.

## D. Hallucination / unsupported-change behavior

A hallucinated change is a concrete physical or structural change that the answer key does not support. The final score of those responses is capped at 3.

The judge marked 19 of 48 scored responses as containing an unsupported structural-change claim.

| Model | Hallucination count | Scored cases |
| --- | --- | --- |
| OPENAI_FRONTIER_VISION | 3 | 6 |
| GEMINI | 3 | 6 |
| CLAUDE | 4 | 6 |
| QWEN | 2 | 6 |
| GEMMA | 1 | 6 |
| LLAMA | 0 | 6 |
| MISTRAL | 1 | 6 |
| KIMI | 5 | 6 |

## E. Representative failure examples

### 1. KIMI on H1

Final score 0 (incorrect); hallucinated_change=True.

The response asserts a physical or structural change that the answer key does not support: The tower architecture changed significantly from an ornate crenellated design with a clock to a simpler pointed spire.

Response:

```text
The tower architecture changed significantly from an ornate crenellated design with a clock in Image B to a simpler pointed spire in Image A. A tall brick chimney visible to the right of the main building in Image A is entirely absent in Image B. The main building's facade, window arrangements, and foreground landscape grading also differ substantially between the two images.
```

### 2. MISTRAL on H2

Final score 1 (incorrect); hallucinated_change=True.

The response asserts a physical or structural change that the answer key does not support: Specific claims about clear and sunny vs. overcast weather.

Response:

```text
The most notable differences between the two images are the environmental and seasonal conditions. In Image A, the weather is clear and sunny, with bright lighting and green vegetation, while in Image B, the sky is overcast, and the trees are bare, indicating a different season, likely winter. Additionally, the flowers in the foreground of Image B are not visible in Image A, suggesting a seasonal change. There are no visible structural or physical changes to the building itself.
```

### 3. GEMINI on M2

Final score 1 (incorrect); hallucinated_change=True.

The response asserts a physical or structural change that the answer key does not support: The monument in Image B features a distinct white band around its lower section that is not present in Image A.

Response:

```text
The most prominent physical difference is the state of the fountain; in Image A, it is filled with water with active spraying jets, whereas in Image B, the basin is completely dry. Additionally, the monument in Image B features a distinct white band around its lower section that is not present in Image A. A major temporary difference is the large crowd of people gathered in the background and foreground of Image B, while Image A is devoid of people.
```

### 4. CLAUDE on M1

Final score 3 (partially_correct); hallucinated_change=True.

The response asserts a physical or structural change that the answer key does not support: The rainbow painting of the stairs represents a permanent structural/decorative change to the building's entrance.

Response:

```text
The most significant change between these two images is that the stairs have been painted with rainbow/pride flag colors in Image B, while they appear to be plain concrete/stone colored in Image A. Additionally, Image B shows snow coverage on the ground and stairs, while Image A shows clear, dry conditions with green foliage. The season has also changed from what appears to be summer/fall (with lush green trees) in Image A to winter (with bare tree branches) in Image B.

The rainbow painting of the stairs represents a permanent structural/decorative change to the building's entrance, while the snow and seasonal vegetation differences are temporary environmental changes.
```

These examples were selected by the rule in `qualitative_failures.md`. They do not represent all behavior.

## F. Interpretation

This six-case pilot suggests that direct VLM comparison is reliable on some clean aligned cases but becomes less consistent when persistent changes must be separated from environmental, historical, or subtle visual differences.

Pooled counts — easy: mean 3.6250 on 16 judgments (14 correct, 1 partial, 1 incorrect, 2 with an unsupported structural claim); medium: mean 3.2500 on 16 judgments (9 correct, 6 partial, 1 incorrect, 5 with an unsupported structural claim); hard: mean 1.4375 on 16 judgments (0 correct, 9 partial, 7 incorrect, 12 with an unsupported structural claim).

The table is a description of this pilot, not a ranking, and it does not estimate performance on other locations or time gaps.

## G. Motivation for MapEvolve

The benchmark gives every VLM the correct pair of images manually.

Therefore failures are downstream visual reasoning failures, not retrieval failures.

A real MapEvolve query is harder because the system must first retrieve and verify the correct historical evidence.

This motivates a pipeline that separates retrieval, pair verification, evidence tracing or refusal, and final VLM generation.

Figures:

- `benchmark/artifacts/model_case_score_matrix.png`
- `benchmark/artifacts/mean_score_by_difficulty.png`
- `benchmark/artifacts/hallucination_rate_by_model.png`
- `benchmark/artifacts/latency_by_model.png`
