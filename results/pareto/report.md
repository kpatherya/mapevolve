# VLM Pareto report (2026-11-14T04:23:03+00:00)

Graded cases with every model: 53. Classifier for the router curve: jev. Membership = share of 300 case bootstraps in which the model is on the frontier.

## all (53 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), unanswerable_quality (max), reliability (max), latency_p90_s (min)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OPENAI_FRONTIER_VISION | 0.881 | 17% | $0.00432 | 5.753 | yes | 100% | — |
| GEMMA4_31B | 0.797 | 11% | $0.00014 | 5.472 | yes | 100% | — |
| GEMMA | 0.797 | 17% | $0.00012 | 8.942 | yes | 100% | — |
| LLAMA | 0.780 | 10% | $0.00150 | 4.717 | yes | 100% | — |
| QWEN3_VL_8B | 0.755 | 23% | $0.00122 | 3.446 | yes | 100% | — |
| MINISTRAL_3B | 0.660 | 23% | $0.00040 | 3.580 | yes | 100% | — |
| GEMMA3_4B | 0.627 | 32% | $0.00005 | 3.849 | yes | 100% | — |
| MISTRAL | 0.551 | 10% | $0.00039 | 4.794 | yes | 99% | — |
| NEMOTRON_VL | 0.802 | 17% | $0.00124 | 6.154 | yes | 89% | — |
| QWEN25_VL_72B | 0.703 | 11% | $0.00464 | 4.857 | yes | 81% | — |
| QWEN | 0.818 | 22% | $0.00124 | 9.513 | yes | 77% | — |
| GLM_4_6V | 0.807 | 15% | $0.00221 | 20.221 | yes | 50% | — |
| QWEN3_VL_30B | 0.684 | 25% | $0.00088 | 8.544 |  | 19% | GEMMA4_31B |
| GEMINI | 0.835 | 22% | $0.02138 | 17.004 |  | 13% | OPENAI_FRONTIER_VISION |

## type=cross_view_matching (14 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), reliability (max), latency_p90_s (min)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 0.750 | 0% | $0.00016 | 8.633 | yes | 100% | — |
| QWEN3_VL_8B | 0.714 | 7% | $0.00144 | 3.761 | yes | 100% | — |
| GEMMA3_4B | 0.554 | 36% | $0.00006 | 3.937 | yes | 100% | — |
| MINISTRAL_3B | 0.446 | 7% | $0.00047 | 4.321 | yes | 100% | — |
| GEMMA | 0.696 | 0% | $0.00015 | 8.187 | yes | 99% | — |
| OPENAI_FRONTIER_VISION | 0.821 | 7% | $0.00526 | 5.635 | yes | 95% | — |
| LLAMA | 0.714 | 0% | $0.00190 | 5.023 | yes | 80% | — |
| NEMOTRON_VL | 0.696 | 0% | $0.00157 | 6.911 | yes | 74% | — |
| QWEN | 0.768 | 14% | $0.00145 | 9.807 | yes | 66% | — |
| GLM_4_6V | 0.804 | 0% | $0.00262 | 22.472 | yes | 58% | — |
| MISTRAL | 0.304 | 14% | $0.00048 | 4.834 |  | 53% | MINISTRAL_3B |
| QWEN25_VL_72B | 0.482 | 0% | $0.00554 | 5.118 | yes | 37% | — |
| GEMINI | 0.768 | 0% | $0.02329 | 17.929 | yes | 33% | — |
| QWEN3_VL_30B | 0.518 | 0% | $0.00110 | 10.000 |  | 23% | GEMMA4_31B |

## type=cross_view_matching|difficulty=easy (2 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 0.750 | 0% | $0.00016 | 12.243 | yes | 100% | — |
| GEMMA3_4B | 0.625 | 0% | $0.00005 | 3.818 | yes | 100% | — |
| QWEN3_VL_8B | 0.625 | 0% | $0.00135 | 3.101 | yes | 100% | — |
| MINISTRAL_3B | 0.500 | 0% | $0.00045 | 3.475 | yes | 100% | — |
| MISTRAL | 0.375 | 50% | $0.00045 | 4.478 | yes | 78% | — |
| GEMINI | 1.000 | 0% | $0.01845 | 15.930 | yes | 71% | — |
| GLM_4_6V | 0.875 | 0% | $0.00235 | 15.660 | yes | 71% | — |
| GEMMA | 0.750 | 0% | $0.00012 | 8.808 | yes | 71% | — |
| QWEN | 0.750 | 0% | $0.00117 | 8.421 | yes | 71% | — |
| NEMOTRON_VL | 0.750 | 0% | $0.00150 | 6.003 | yes | 49% | — |
| LLAMA | 0.625 | 0% | $0.00185 | 4.223 |  | 22% | QWEN3_VL_8B |
| QWEN3_VL_30B | 0.500 | 0% | $0.00116 | 7.726 |  | 22% | GEMMA3_4B, MINISTRAL_3B |
| OPENAI_FRONTIER_VISION | 0.625 | 0% | $0.00445 | 5.117 |  | 0% | LLAMA, QWEN3_VL_8B |
| QWEN25_VL_72B | 0.500 | 0% | $0.00528 | 4.692 |  | 0% | GEMMA3_4B, LLAMA, MINISTRAL_3B, QWEN3_VL_8B |

## type=cross_view_matching|difficulty=hard (7 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 0.571 | 0% | $0.00018 | 5.953 | yes | 100% | — |
| GEMMA3_4B | 0.286 | 57% | $0.00006 | 4.237 | yes | 100% | — |
| OPENAI_FRONTIER_VISION | 0.786 | 14% | $0.00566 | 6.320 | yes | 99% | — |
| GEMMA | 0.464 | 0% | $0.00015 | 6.980 | yes | 97% | — |
| QWEN3_VL_8B | 0.571 | 0% | $0.00147 | 4.339 | yes | 94% | — |
| MINISTRAL_3B | 0.286 | 0% | $0.00048 | 4.302 | yes | 93% | — |
| NEMOTRON_VL | 0.679 | 0% | $0.00148 | 7.294 | yes | 84% | — |
| QWEN | 0.643 | 14% | $0.00155 | 8.590 |  | 39% | NEMOTRON_VL |
| MISTRAL | 0.179 | 14% | $0.00049 | 4.539 |  | 35% | MINISTRAL_3B |
| LLAMA | 0.536 | 0% | $0.00193 | 5.249 |  | 32% | QWEN3_VL_8B |
| GLM_4_6V | 0.679 | 0% | $0.00272 | 24.804 |  | 29% | NEMOTRON_VL |
| GEMINI | 0.571 | 0% | $0.02587 | 19.372 |  | 8% | GEMMA4_31B, NEMOTRON_VL, QWEN3_VL_8B |
| QWEN3_VL_30B | 0.393 | 0% | $0.00091 | 8.773 |  | 3% | GEMMA4_31B |
| QWEN25_VL_72B | 0.357 | 0% | $0.00566 | 5.570 |  | 0% | LLAMA, QWEN3_VL_8B |

## type=cross_view_matching|difficulty=moderate (5 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 1.000 | 0% | $0.00015 | 10.942 | yes | 100% | — |
| QWEN3_VL_8B | 0.950 | 20% | $0.00143 | 3.216 | yes | 100% | — |
| GEMMA3_4B | 0.900 | 20% | $0.00006 | 3.564 | yes | 100% | — |
| MINISTRAL_3B | 0.650 | 20% | $0.00046 | 4.686 | yes | 99% | — |
| MISTRAL | 0.450 | 0% | $0.00047 | 5.389 | yes | 75% | — |
| GEMMA | 1.000 | 0% | $0.00015 | 9.627 | yes | 74% | — |
| LLAMA | 1.000 | 0% | $0.00188 | 5.027 | yes | 68% | — |
| QWEN25_VL_72B | 0.650 | 0% | $0.00549 | 4.657 | yes | 58% | — |
| NEMOTRON_VL | 0.700 | 0% | $0.00171 | 6.738 | yes | 47% | — |
| OPENAI_FRONTIER_VISION | 0.950 | 0% | $0.00504 | 4.884 | yes | 37% | — |
| QWEN3_VL_30B | 0.700 | 0% | $0.00135 | 12.627 |  | 23% | GEMMA4_31B |
| QWEN | 0.950 | 20% | $0.00144 | 12.065 |  | 1% | GEMMA4_31B, QWEN3_VL_8B |
| GEMINI | 0.950 | 0% | $0.02162 | 16.708 |  | 0% | GEMMA4_31B, LLAMA, OPENAI_FRONTIER_VISION |
| GLM_4_6V | 0.950 | 0% | $0.00257 | 21.933 |  | 0% | GEMMA4_31B, LLAMA |

## type=factual_retrieval (6 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), unanswerable_quality (max), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 1.000 | 0% | $0.00009 | 3.442 | yes | 100% | — |
| GEMMA3_4B | 0.667 | 33% | $0.00004 | 6.624 | yes | 100% | — |
| QWEN3_VL_8B | 0.833 | 17% | $0.00080 | 2.349 | yes | 87% | — |
| MINISTRAL_3B | 0.917 | 33% | $0.00026 | 2.951 | yes | 78% | — |
| NEMOTRON_VL | 1.000 | 0% | $0.00069 | 2.925 | yes | 66% | — |
| GEMMA | 0.917 | 0% | $0.00010 | 6.460 |  | 30% | GEMMA4_31B |
| MISTRAL | 0.833 | 17% | $0.00025 | 3.590 |  | 29% | GEMMA4_31B |
| LLAMA | 1.000 | 0% | $0.00099 | 3.054 |  | 15% | NEMOTRON_VL |
| QWEN25_VL_72B | 0.875 | 17% | $0.00298 | 3.536 |  | 5% | GEMMA4_31B, LLAMA, NEMOTRON_VL, OPENAI_FRONTIER_VISION |
| OPENAI_FRONTIER_VISION | 1.000 | 0% | $0.00291 | 3.295 |  | 3% | LLAMA, NEMOTRON_VL |
| GEMINI | 1.000 | 0% | $0.01320 | 10.994 |  | 0% | GEMMA4_31B, LLAMA, NEMOTRON_VL, OPENAI_FRONTIER_VISION, QWEN |
| QWEN | 1.000 | 0% | $0.00093 | 4.419 |  | 0% | GEMMA4_31B, NEMOTRON_VL |
| GLM_4_6V | 0.917 | 0% | $0.00134 | 10.797 |  | 0% | GEMMA4_31B, LLAMA, NEMOTRON_VL, QWEN |
| QWEN3_VL_30B | 0.875 | 17% | $0.00057 | 7.242 |  | 0% | GEMMA4_31B |

## type=factual_retrieval|difficulty=easy (2 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 1.000 | 0% | $0.00009 | 2.911 | yes | 100% | — |
| NEMOTRON_VL | 1.000 | 0% | $0.00058 | 1.865 | yes | 100% | — |
| MINISTRAL_3B | 0.875 | 50% | $0.00024 | 2.398 | yes | 100% | — |
| GEMMA3_4B | 0.625 | 50% | $0.00004 | 3.971 | yes | 100% | — |
| MISTRAL | 0.875 | 0% | $0.00022 | 2.860 | yes | 71% | — |
| QWEN3_VL_8B | 1.000 | 0% | $0.00073 | 2.054 |  | 29% | NEMOTRON_VL |
| GEMINI | 1.000 | 0% | $0.01188 | 9.123 |  | 0% | GEMMA4_31B, GLM_4_6V, LLAMA, NEMOTRON_VL, OPENAI_FRONTIER_VISION, QWEN, QWEN3_VL_8B |
| GLM_4_6V | 1.000 | 0% | $0.00114 | 6.883 |  | 0% | GEMMA4_31B, LLAMA, NEMOTRON_VL, QWEN, QWEN3_VL_8B |
| LLAMA | 1.000 | 0% | $0.00096 | 2.675 |  | 0% | NEMOTRON_VL, QWEN3_VL_8B |
| OPENAI_FRONTIER_VISION | 1.000 | 0% | $0.00272 | 2.758 |  | 0% | LLAMA, NEMOTRON_VL, QWEN3_VL_8B |
| QWEN | 1.000 | 0% | $0.00078 | 4.620 |  | 0% | GEMMA4_31B, NEMOTRON_VL, QWEN3_VL_8B |
| QWEN25_VL_72B | 1.000 | 0% | $0.00266 | 2.926 |  | 0% | GEMMA4_31B, LLAMA, NEMOTRON_VL, QWEN3_VL_8B |
| QWEN3_VL_30B | 1.000 | 0% | $0.00063 | 7.495 |  | 0% | GEMMA4_31B, NEMOTRON_VL |
| GEMMA | 0.875 | 0% | $0.00009 | 6.159 |  | 0% | GEMMA4_31B |

## type=factual_retrieval|difficulty=hard (3 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), unanswerable_quality (max), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 1.000 | 0% | $0.00009 | 3.930 | yes | 100% | — |
| GEMMA3_4B | 0.833 | 0% | $0.00004 | 9.814 | yes | 100% | — |
| QWEN3_VL_8B | 0.667 | 33% | $0.00085 | 2.596 | yes | 93% | — |
| MINISTRAL_3B | 1.000 | 0% | $0.00028 | 3.347 | yes | 76% | — |
| GEMMA | 1.000 | 0% | $0.00009 | 6.543 |  | 30% | GEMMA4_31B |
| QWEN25_VL_72B | 1.000 | 0% | $0.00326 | 3.312 | yes | 20% | — |
| NEMOTRON_VL | 1.000 | 0% | $0.00076 | 3.846 |  | 16% | MINISTRAL_3B |
| OPENAI_FRONTIER_VISION | 1.000 | 0% | $0.00284 | 3.395 |  | 14% | MINISTRAL_3B |
| LLAMA | 1.000 | 0% | $0.00103 | 3.454 |  | 10% | MINISTRAL_3B |
| GEMINI | 1.000 | 0% | $0.01345 | 11.325 |  | 0% | GEMMA4_31B, LLAMA, MINISTRAL_3B, NEMOTRON_VL, OPENAI_FRONTIER_VISION, QWEN |
| QWEN | 1.000 | 0% | $0.00102 | 4.547 |  | 0% | GEMMA4_31B, MINISTRAL_3B, NEMOTRON_VL |
| QWEN3_VL_30B | 1.000 | 0% | $0.00056 | 7.966 |  | 0% | GEMMA4_31B, MINISTRAL_3B |
| GLM_4_6V | 0.833 | 0% | $0.00151 | 13.486 |  | 0% | GEMMA4_31B, LLAMA, MINISTRAL_3B, NEMOTRON_VL, QWEN |
| MISTRAL | 0.750 | 33% | $0.00028 | 4.085 |  | 0% | GEMMA4_31B, MINISTRAL_3B |

## type=landmark_identification (9 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), unanswerable_quality (max), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 0.833 | 0% | $0.00010 | 4.845 | yes | 100% | — |
| GEMMA3_4B | 0.694 | 0% | $0.00004 | 2.995 | yes | 100% | — |
| MINISTRAL_3B | 0.806 | 0% | $0.00025 | 2.543 | yes | 100% | — |
| GEMMA | 0.944 | 0% | $0.00009 | 10.390 | yes | 99% | — |
| MISTRAL | 0.750 | 0% | $0.00022 | 3.033 | yes | 98% | — |
| QWEN3_VL_8B | 0.889 | 0% | $0.00078 | 2.898 | yes | 88% | — |
| OPENAI_FRONTIER_VISION | 1.000 | 0% | $0.00336 | 4.230 | yes | 82% | — |
| QWEN3_VL_30B | 0.889 | 0% | $0.00051 | 4.815 | yes | 76% | — |
| QWEN | 0.944 | 0% | $0.00082 | 11.047 | yes | 66% | — |
| LLAMA | 0.833 | 0% | $0.00101 | 4.051 |  | 24% | QWEN3_VL_8B |
| NEMOTRON_VL | 0.833 | 0% | $0.00090 | 6.316 |  | 3% | GEMMA4_31B, QWEN3_VL_8B |
| GEMINI | 0.944 | 0% | $0.01768 | 13.976 |  | 0% | OPENAI_FRONTIER_VISION, QWEN |
| GLM_4_6V | 0.889 | 0% | $0.00132 | 14.211 |  | 0% | QWEN, QWEN3_VL_8B |
| QWEN25_VL_72B | 0.861 | 0% | $0.00280 | 4.284 |  | 0% | QWEN3_VL_8B |

## type=landmark_identification|difficulty=easy (3 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA3_4B | 1.000 | 0% | $0.00004 | 2.696 | yes | 100% | — |
| GEMMA4_31B | 0.833 | 0% | $0.00011 | 3.798 | yes | 100% | — |
| QWEN3_VL_8B | 1.000 | 0% | $0.00072 | 2.439 | yes | 96% | — |
| MINISTRAL_3B | 0.917 | 0% | $0.00022 | 2.456 | yes | 96% | — |
| MISTRAL | 0.917 | 0% | $0.00020 | 3.005 | yes | 79% | — |
| QWEN | 1.000 | 0% | $0.00072 | 16.715 | yes | 38% | — |
| GEMINI | 1.000 | 0% | $0.01580 | 11.626 |  | 0% | NEMOTRON_VL, OPENAI_FRONTIER_VISION, QWEN3_VL_8B |
| GEMMA | 1.000 | 0% | $0.00009 | 13.104 |  | 0% | GEMMA3_4B |
| GLM_4_6V | 1.000 | 0% | $0.00118 | 13.872 |  | 0% | NEMOTRON_VL, QWEN3_VL_8B |
| NEMOTRON_VL | 1.000 | 0% | $0.00094 | 3.169 |  | 0% | QWEN3_VL_8B |
| OPENAI_FRONTIER_VISION | 1.000 | 0% | $0.00309 | 3.686 |  | 0% | NEMOTRON_VL, QWEN3_VL_8B |
| QWEN3_VL_30B | 1.000 | 0% | $0.00055 | 7.190 |  | 0% | GEMMA3_4B |
| QWEN25_VL_72B | 0.917 | 0% | $0.00255 | 3.504 |  | 0% | GEMMA3_4B, MINISTRAL_3B, MISTRAL, NEMOTRON_VL, QWEN3_VL_8B |
| LLAMA | 0.667 | 0% | $0.00096 | 4.049 |  | 0% | GEMMA4_31B, MINISTRAL_3B, MISTRAL, NEMOTRON_VL, QWEN3_VL_8B |

## type=landmark_identification|difficulty=hard (4 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), unanswerable_quality (max), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 0.750 | 0% | $0.00010 | 4.569 | yes | 100% | — |
| GEMMA3_4B | 0.500 | 0% | $0.00005 | 2.101 | yes | 100% | — |
| GEMMA | 0.875 | 0% | $0.00009 | 11.042 | yes | 94% | — |
| MINISTRAL_3B | 0.750 | 0% | $0.00027 | 2.562 | yes | 92% | — |
| MISTRAL | 0.750 | 0% | $0.00023 | 3.265 | yes | 79% | — |
| OPENAI_FRONTIER_VISION | 1.000 | 0% | $0.00373 | 4.776 | yes | 71% | — |
| LLAMA | 0.875 | 0% | $0.00106 | 3.825 | yes | 65% | — |
| QWEN | 0.875 | 0% | $0.00093 | 9.733 | yes | 65% | — |
| QWEN3_VL_8B | 0.750 | 0% | $0.00083 | 3.284 |  | 7% | MINISTRAL_3B, MISTRAL |
| GEMINI | 0.875 | 0% | $0.01959 | 15.264 |  | 0% | LLAMA, OPENAI_FRONTIER_VISION, QWEN |
| GLM_4_6V | 0.750 | 0% | $0.00146 | 16.865 |  | 0% | GEMMA4_31B, LLAMA, MINISTRAL_3B, MISTRAL, QWEN, QWEN3_VL_8B |
| QWEN25_VL_72B | 0.750 | 0% | $0.00301 | 4.214 |  | 0% | LLAMA, MINISTRAL_3B, MISTRAL, QWEN3_VL_30B, QWEN3_VL_8B |
| QWEN3_VL_30B | 0.750 | 0% | $0.00052 | 3.775 |  | 0% | MINISTRAL_3B, MISTRAL |
| NEMOTRON_VL | 0.625 | 0% | $0.00093 | 10.096 |  | 0% | GEMMA4_31B, MINISTRAL_3B, MISTRAL, QWEN, QWEN3_VL_8B |

## type=landmark_identification|difficulty=moderate (2 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 1.000 | 0% | $0.00011 | 6.965 | yes | 100% | — |
| GEMMA3_4B | 0.625 | 0% | $0.00003 | 5.231 | yes | 100% | — |
| GEMMA | 1.000 | 0% | $0.00007 | 5.015 | yes | 78% | — |
| MISTRAL | 0.500 | 0% | $0.00022 | 2.612 | yes | 78% | — |
| MINISTRAL_3B | 0.750 | 0% | $0.00025 | 2.635 | yes | 71% | — |
| QWEN | 1.000 | 0% | $0.00075 | 5.173 | yes | 49% | — |
| QWEN3_VL_30B | 1.000 | 0% | $0.00046 | 3.334 | yes | 49% | — |
| QWEN3_VL_8B | 1.000 | 0% | $0.00076 | 2.814 | yes | 49% | — |
| NEMOTRON_VL | 1.000 | 0% | $0.00078 | 3.477 |  | 22% | QWEN3_VL_8B |
| GEMINI | 1.000 | 0% | $0.01667 | 14.925 |  | 0% | GEMMA4_31B, GLM_4_6V, LLAMA, NEMOTRON_VL, OPENAI_FRONTIER_VISION, QWEN, QWEN3_VL_8B |
| GLM_4_6V | 1.000 | 0% | $0.00124 | 9.415 |  | 0% | GEMMA4_31B, LLAMA, NEMOTRON_VL, QWEN, QWEN3_VL_8B |
| LLAMA | 1.000 | 0% | $0.00100 | 4.503 |  | 0% | NEMOTRON_VL, QWEN3_VL_8B |
| OPENAI_FRONTIER_VISION | 1.000 | 0% | $0.00306 | 3.951 |  | 0% | NEMOTRON_VL, QWEN3_VL_8B |
| QWEN25_VL_72B | 1.000 | 0% | $0.00273 | 5.593 |  | 0% | LLAMA, NEMOTRON_VL, QWEN, QWEN3_VL_30B, QWEN3_VL_8B |

## type=spatial_reasoning (6 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), unanswerable_quality (max), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 0.875 | 0% | $0.00013 | 2.930 | yes | 100% | — |
| GEMMA3_4B | 0.833 | 33% | $0.00005 | 3.668 | yes | 100% | — |
| GEMMA | 0.875 | 17% | $0.00013 | 5.449 | yes | 84% | — |
| MINISTRAL_3B | 0.792 | 17% | $0.00040 | 2.938 | yes | 79% | — |
| NEMOTRON_VL | 0.958 | 17% | $0.00103 | 3.159 | yes | 64% | — |
| LLAMA | 0.917 | 0% | $0.00168 | 4.089 | yes | 45% | — |
| QWEN | 0.917 | 0% | $0.00113 | 5.316 | yes | 42% | — |
| OPENAI_FRONTIER_VISION | 1.000 | 0% | $0.00435 | 4.404 | yes | 42% | — |
| MISTRAL | 0.542 | 17% | $0.00039 | 3.468 |  | 23% | GEMMA4_31B |
| QWEN3_VL_8B | 0.833 | 17% | $0.00122 | 4.758 |  | 20% | GEMMA4_31B, NEMOTRON_VL |
| GLM_4_6V | 0.958 | 0% | $0.00197 | 17.160 | yes | 14% | — |
| QWEN3_VL_30B | 0.750 | 17% | $0.00106 | 11.953 |  | 7% | GEMMA4_31B, MINISTRAL_3B, NEMOTRON_VL |
| QWEN25_VL_72B | 0.750 | 17% | $0.00468 | 4.839 |  | 4% | GEMMA4_31B, LLAMA, MINISTRAL_3B, NEMOTRON_VL, OPENAI_FRONTIER_VISION, QWEN3_VL_8B |
| GEMINI | 0.958 | 17% | $0.01678 | 11.607 |  | 0% | NEMOTRON_VL, OPENAI_FRONTIER_VISION |

## type=spatial_reasoning|difficulty=hard (3 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), unanswerable_quality (max), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 0.917 | 0% | $0.00014 | 2.945 | yes | 100% | — |
| GEMMA3_4B | 0.833 | 33% | $0.00005 | 4.806 | yes | 100% | — |
| GEMMA | 1.000 | 0% | $0.00012 | 5.830 | yes | 97% | — |
| NEMOTRON_VL | 1.000 | 0% | $0.00110 | 2.876 | yes | 96% | — |
| MINISTRAL_3B | 1.000 | 0% | $0.00045 | 3.237 | yes | 67% | — |
| GEMINI | 1.000 | 0% | $0.01797 | 12.663 |  | 0% | MINISTRAL_3B, NEMOTRON_VL, OPENAI_FRONTIER_VISION, QWEN |
| GLM_4_6V | 1.000 | 0% | $0.00230 | 20.050 |  | 0% | MINISTRAL_3B, NEMOTRON_VL, QWEN |
| OPENAI_FRONTIER_VISION | 1.000 | 0% | $0.00442 | 4.270 |  | 0% | MINISTRAL_3B, NEMOTRON_VL, QWEN |
| QWEN | 1.000 | 0% | $0.00128 | 3.523 |  | 0% | MINISTRAL_3B, NEMOTRON_VL |
| LLAMA | 0.917 | 0% | $0.00181 | 3.702 |  | 0% | GEMMA4_31B, MINISTRAL_3B, NEMOTRON_VL, QWEN |
| QWEN25_VL_72B | 0.667 | 33% | $0.00544 | 4.908 |  | 0% | GEMMA3_4B, GEMMA4_31B, LLAMA, MINISTRAL_3B, NEMOTRON_VL, OPENAI_FRONTIER_VISION, QWEN, QWEN3_VL_8B |
| QWEN3_VL_30B | 0.667 | 33% | $0.00112 | 12.471 |  | 0% | GEMMA3_4B, GEMMA4_31B, MINISTRAL_3B, NEMOTRON_VL |
| QWEN3_VL_8B | 0.667 | 33% | $0.00140 | 3.441 |  | 0% | GEMMA4_31B, MINISTRAL_3B, NEMOTRON_VL |
| MISTRAL | 0.583 | 33% | $0.00046 | 4.213 |  | 0% | GEMMA4_31B, MINISTRAL_3B |

## type=spatial_reasoning|difficulty=moderate (2 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA3_4B | 0.750 | 50% | $0.00006 | 2.631 | yes | 100% | — |
| GEMMA4_31B | 0.750 | 0% | $0.00014 | 3.699 | yes | 100% | — |
| MINISTRAL_3B | 0.500 | 0% | $0.00043 | 2.957 | yes | 71% | — |
| NEMOTRON_VL | 1.000 | 0% | $0.00120 | 4.229 | yes | 49% | — |
| MISTRAL | 0.750 | 0% | $0.00039 | 3.025 | yes | 49% | — |
| GEMMA | 0.750 | 0% | $0.00015 | 5.667 |  | 22% | GEMMA4_31B |
| OPENAI_FRONTIER_VISION | 1.000 | 0% | $0.00492 | 5.266 |  | 0% | NEMOTRON_VL |
| QWEN3_VL_8B | 1.000 | 0% | $0.00129 | 8.148 |  | 0% | NEMOTRON_VL |
| GEMINI | 0.875 | 50% | $0.01703 | 11.361 |  | 0% | LLAMA, NEMOTRON_VL, OPENAI_FRONTIER_VISION, QWEN3_VL_8B |
| GLM_4_6V | 0.875 | 0% | $0.00203 | 14.721 |  | 0% | LLAMA, NEMOTRON_VL, QWEN3_VL_8B |
| LLAMA | 0.875 | 0% | $0.00184 | 5.631 |  | 0% | NEMOTRON_VL |
| QWEN | 0.750 | 0% | $0.00117 | 9.383 |  | 0% | GEMMA4_31B, MISTRAL |
| QWEN25_VL_72B | 0.750 | 0% | $0.00492 | 5.181 |  | 0% | GEMMA4_31B, MISTRAL, NEMOTRON_VL |
| QWEN3_VL_30B | 0.750 | 0% | $0.00119 | 12.386 |  | 0% | GEMMA4_31B, MISTRAL, QWEN |

## type=temporal_comparison (18 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), reliability (max), latency_p90_s (min)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA | 0.750 | 38% | $0.00013 | 10.334 | yes | 100% | — |
| GEMMA4_31B | 0.722 | 33% | $0.00015 | 4.850 | yes | 100% | — |
| QWEN3_VL_8B | 0.667 | 50% | $0.00142 | 3.403 | yes | 100% | — |
| GEMMA3_4B | 0.569 | 44% | $0.00006 | 3.345 | yes | 100% | — |
| MISTRAL | 0.552 | 8% | $0.00044 | 6.064 | yes | 100% | — |
| MINISTRAL_3B | 0.625 | 44% | $0.00047 | 3.945 | yes | 96% | — |
| QWEN25_VL_72B | 0.722 | 22% | $0.00539 | 5.388 | yes | 87% | — |
| OPENAI_FRONTIER_VISION | 0.812 | 38% | $0.00446 | 7.346 | yes | 83% | — |
| LLAMA | 0.708 | 25% | $0.00152 | 5.361 | yes | 82% | — |
| NEMOTRON_VL | 0.750 | 44% | $0.00141 | 7.558 | yes | 57% | — |
| QWEN | 0.729 | 46% | $0.00139 | 11.089 | yes | 50% | — |
| GLM_4_6V | 0.681 | 44% | $0.00271 | 25.635 |  | 17% | GEMMA4_31B, LLAMA, NEMOTRON_VL |
| GEMINI | 0.760 | 50% | $0.02486 | 20.453 |  | 15% | OPENAI_FRONTIER_VISION |
| QWEN3_VL_30B | 0.625 | 61% | $0.00093 | 8.574 |  | 6% | GEMMA4_31B, MINISTRAL_3B |

## type=temporal_comparison|difficulty=easy (4 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA3_4B | 0.938 | 0% | $0.00006 | 3.122 | yes | 100% | — |
| GEMMA4_31B | 0.938 | 25% | $0.00014 | 3.853 | yes | 100% | — |
| LLAMA | 1.000 | 0% | $0.00148 | 6.341 | yes | 71% | — |
| MISTRAL | 0.917 | 0% | $0.00045 | 6.987 | yes | 71% | — |
| QWEN3_VL_8B | 0.625 | 50% | $0.00152 | 3.780 | yes | 57% | — |
| MINISTRAL_3B | 0.812 | 25% | $0.00049 | 4.777 |  | 50% | GEMMA4_31B |
| OPENAI_FRONTIER_VISION | 0.917 | 17% | $0.00416 | 5.155 | yes | 34% | — |
| QWEN | 0.958 | 17% | $0.00150 | 12.518 |  | 23% | LLAMA |
| GEMMA | 0.917 | 17% | $0.00011 | 10.804 |  | 23% | GEMMA3_4B |
| NEMOTRON_VL | 0.750 | 50% | $0.00157 | 7.752 |  | 3% | GEMMA4_31B, LLAMA, MINISTRAL_3B, MISTRAL |
| GEMINI | 0.875 | 17% | $0.02507 | 19.148 |  | 0% | LLAMA, MISTRAL, OPENAI_FRONTIER_VISION, QWEN |
| QWEN25_VL_72B | 0.812 | 25% | $0.00580 | 4.896 |  | 0% | GEMMA3_4B, GEMMA4_31B, MINISTRAL_3B |
| GLM_4_6V | 0.562 | 75% | $0.00318 | 33.887 |  | 0% | GEMMA4_31B, LLAMA, MINISTRAL_3B, MISTRAL, NEMOTRON_VL, QWEN, QWEN3_VL_8B |
| QWEN3_VL_30B | 0.562 | 75% | $0.00094 | 8.637 |  | 0% | GEMMA3_4B, GEMMA4_31B, MINISTRAL_3B, MISTRAL |

## type=temporal_comparison|difficulty=hard (6 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA4_31B | 0.458 | 33% | $0.00015 | 6.026 | yes | 100% | — |
| GEMMA3_4B | 0.333 | 67% | $0.00006 | 3.174 | yes | 100% | — |
| MISTRAL | 0.344 | 25% | $0.00038 | 4.890 | yes | 99% | — |
| QWEN3_VL_8B | 0.458 | 83% | $0.00121 | 3.053 | yes | 96% | — |
| MINISTRAL_3B | 0.375 | 67% | $0.00042 | 3.672 | yes | 96% | — |
| OPENAI_FRONTIER_VISION | 0.625 | 62% | $0.00467 | 8.996 | yes | 77% | — |
| GEMMA | 0.406 | 75% | $0.00014 | 7.870 | yes | 69% | — |
| QWEN25_VL_72B | 0.458 | 50% | $0.00449 | 4.690 | yes | 59% | — |
| NEMOTRON_VL | 0.500 | 67% | $0.00142 | 9.455 | yes | 47% | — |
| LLAMA | 0.312 | 50% | $0.00138 | 4.793 | yes | 43% | — |
| GLM_4_6V | 0.500 | 67% | $0.00228 | 20.637 |  | 40% | NEMOTRON_VL |
| GEMINI | 0.625 | 75% | $0.02483 | 21.519 |  | 37% | OPENAI_FRONTIER_VISION |
| QWEN | 0.406 | 75% | $0.00117 | 9.402 |  | 23% | GEMMA4_31B |
| QWEN3_VL_30B | 0.375 | 83% | $0.00083 | 9.839 |  | 1% | GEMMA4_31B, MINISTRAL_3B |

## type=temporal_comparison|difficulty=moderate (8 cases)

Objectives: quality (max), unsupported_rate (min), cost_usd (min), latency_s (min), reliability (max)

| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMMA | 0.925 | 20% | $0.00013 | 12.024 | yes | 100% | — |
| GEMMA4_31B | 0.812 | 38% | $0.00016 | 4.466 | yes | 100% | — |
| GEMMA3_4B | 0.562 | 50% | $0.00006 | 3.584 | yes | 100% | — |
| QWEN3_VL_8B | 0.844 | 25% | $0.00153 | 3.478 | yes | 97% | — |
| NEMOTRON_VL | 0.938 | 25% | $0.00132 | 6.038 | yes | 95% | — |
| MISTRAL | 0.500 | 0% | $0.00048 | 6.449 | yes | 94% | — |
| MINISTRAL_3B | 0.719 | 38% | $0.00050 | 3.734 | yes | 92% | — |
| QWEN25_VL_72B | 0.875 | 0% | $0.00587 | 6.157 | yes | 85% | — |
| LLAMA | 0.850 | 20% | $0.00166 | 5.228 | yes | 71% | — |
| GLM_4_6V | 0.875 | 12% | $0.00280 | 25.258 | yes | 55% | — |
| QWEN3_VL_30B | 0.844 | 38% | $0.00101 | 7.593 | yes | 55% | — |
| OPENAI_FRONTIER_VISION | 0.900 | 30% | $0.00447 | 7.341 |  | 46% | NEMOTRON_VL |
| QWEN | 0.850 | 40% | $0.00149 | 11.582 |  | 17% | NEMOTRON_VL |
| GEMINI | 0.800 | 50% | $0.02475 | 20.382 |  | 2% | GEMMA4_31B, LLAMA, NEMOTRON_VL, OPENAI_FRONTIER_VISION, QWEN, QWEN3_VL_8B |

## Router frontier (mode × budget × weights, leave-one-case-out)

| Setting | Quality | Cost/query |
| --- | --- | --- |
| balanced budget=$0.0002 w=0.4/0.4/0.2 | 0.542 | $0.00006 |
| balanced budget=$0.0002 w=0.8/0.1/0.1 | 0.625 | $0.00009 |
| balanced budget=$0.005 w=0.6/0.25/0.15 | 0.708 | $0.00012 |
| economy budget=$0.0002 | 0.750 | $0.00013 |
