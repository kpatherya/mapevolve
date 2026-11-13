# VLM Pareto report (2026-11-13T04:07:47+00:00)

Graded cases with every model: 6. Classifier for the router curve: heuristic. Membership = share of 300 case bootstraps in which the model is on the frontier.

## all (6 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA | 0.750 | 17% | $0.00009 | 4.932 | yes | 100% | — |
| MISTRAL | 0.792 | 17% | $0.00036 | 7.982 | yes | 60% | — |
| LLAMA | 0.667 | 0% | $0.00079 | 7.618 | yes | 57% | — |
| CLAUDE | 0.667 | 67% | $0.01062 | 5.894 |  | 24% | GEMMA |
| OPENAI_FRONTIER_VISION | 0.750 | 50% | $0.00295 | 12.286 |  | 22% | GEMMA, MISTRAL |
| GEMINI | 0.708 | 50% | $0.02533 | 19.652 |  | 2% | GEMMA, MISTRAL, OPENAI_FRONTIER_VISION, QWEN |
| QWEN | 0.750 | 33% | $0.00098 | 16.724 |  | 0% | GEMMA, MISTRAL |
| KIMI | 0.458 | 83% | $0.01159 | 74.263 |  | 0% | CLAUDE, GEMMA, LLAMA, MISTRAL, OPENAI_FRONTIER_VISION, QWEN |

## type=temporal_comparison (6 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA | 0.750 | 17% | $0.00009 | 4.932 | yes | 100% | — |
| MISTRAL | 0.792 | 17% | $0.00036 | 7.982 | yes | 60% | — |
| LLAMA | 0.667 | 0% | $0.00079 | 7.618 | yes | 57% | — |
| CLAUDE | 0.667 | 67% | $0.01062 | 5.894 |  | 24% | GEMMA |
| OPENAI_FRONTIER_VISION | 0.750 | 50% | $0.00295 | 12.286 |  | 22% | GEMMA, MISTRAL |
| GEMINI | 0.708 | 50% | $0.02533 | 19.652 |  | 2% | GEMMA, MISTRAL, OPENAI_FRONTIER_VISION, QWEN |
| QWEN | 0.750 | 33% | $0.00098 | 16.724 |  | 0% | GEMMA, MISTRAL |
| KIMI | 0.458 | 83% | $0.01159 | 74.263 |  | 0% | CLAUDE, GEMMA, LLAMA, MISTRAL, OPENAI_FRONTIER_VISION, QWEN |

## type=temporal_comparison|difficulty=easy (2 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA | 1.000 | 0% | $0.00009 | 4.687 | yes | 100% | — |
| OPENAI_FRONTIER_VISION | 1.000 | 0% | $0.00297 | 4.080 | yes | 71% | — |
| MISTRAL | 1.000 | 0% | $0.00036 | 10.662 |  | 22% | GEMMA |
| GEMINI | 1.000 | 0% | $0.02476 | 20.679 |  | 0% | GEMMA, LLAMA, MISTRAL, OPENAI_FRONTIER_VISION |
| LLAMA | 1.000 | 0% | $0.00087 | 9.630 |  | 0% | GEMMA |
| QWEN | 1.000 | 0% | $0.00127 | 25.777 |  | 0% | GEMMA, LLAMA, MISTRAL |
| CLAUDE | 0.750 | 50% | $0.01201 | 6.725 |  | 0% | GEMMA, OPENAI_FRONTIER_VISION |
| KIMI | 0.500 | 50% | $0.01290 | 104.650 |  | 0% | CLAUDE, GEMMA, LLAMA, MISTRAL, OPENAI_FRONTIER_VISION, QWEN |

## type=temporal_comparison|difficulty=hard (2 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA | 0.375 | 50% | $0.00009 | 4.841 | yes | 100% | — |
| OPENAI_FRONTIER_VISION | 0.500 | 100% | $0.00311 | 18.188 | yes | 78% | — |
| LLAMA | 0.250 | 0% | $0.00061 | 6.127 | yes | 49% | — |
| CLAUDE | 0.375 | 100% | $0.00825 | 6.242 |  | 22% | GEMMA |
| MISTRAL | 0.375 | 50% | $0.00026 | 6.544 |  | 22% | GEMMA |
| GEMINI | 0.500 | 100% | $0.02611 | 19.273 |  | 0% | OPENAI_FRONTIER_VISION |
| KIMI | 0.250 | 100% | $0.00980 | 58.529 |  | 0% | CLAUDE, GEMMA, LLAMA, MISTRAL, OPENAI_FRONTIER_VISION, QWEN |
| QWEN | 0.250 | 100% | $0.00066 | 10.594 |  | 0% | GEMMA, LLAMA, MISTRAL |

## type=temporal_comparison|difficulty=moderate (2 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLAUDE | 0.875 | 50% | $0.01159 | 4.714 | yes | 100% | — |
| GEMMA | 0.875 | 0% | $0.00010 | 5.267 | yes | 100% | — |
| MISTRAL | 1.000 | 0% | $0.00046 | 6.739 | yes | 78% | — |
| QWEN | 1.000 | 0% | $0.00102 | 13.800 |  | 0% | MISTRAL |
| LLAMA | 0.750 | 0% | $0.00089 | 7.097 |  | 0% | GEMMA, MISTRAL |
| OPENAI_FRONTIER_VISION | 0.750 | 50% | $0.00279 | 14.590 |  | 0% | GEMMA, LLAMA, MISTRAL, QWEN |
| GEMINI | 0.625 | 50% | $0.02511 | 19.003 |  | 0% | CLAUDE, GEMMA, LLAMA, MISTRAL, OPENAI_FRONTIER_VISION, QWEN |
| KIMI | 0.625 | 100% | $0.01208 | 59.609 |  | 0% | CLAUDE, GEMMA, LLAMA, MISTRAL, OPENAI_FRONTIER_VISION, QWEN |

## Router frontier (mode × budget × weights, leave-one-case-out)

| Setting | Quality | Cost/query |
| --- | --- | --- |
| economy budget=$0.0002 | 0.750 | $0.00009 |
