# Router tuning on dev (31 cases)

Classifier: jev. Registry built from dev grades only, leave-one-case-out.

| Mode | Configs | Default score /4 | Default cost | Tuned score /4 | Tuned cost | Configs tied with best |
| --- | --- | --- | --- | --- | --- | --- |
| economy | 100 | 3.0645 | $0.000952 | 3.2258 | $0.0002557 | 1 |
| balanced | 80 | 2.7742 | $0.000214 | 3.2903 | $0.0024091 | 1 |
| accuracy | 20 | 3.4516 | $0.004141 | 3.4516 | $0.0039236 | 4 |

Single-model baselines on the same dev cases:

| Strategy | Score /4 | Cost |
| --- | --- | --- |
| always GEMINI | 3.1935 | $0.0209206 |
| always GEMMA | 3.0968 | $0.0001291 |
| always GEMMA3_4B | 2.4194 | $5.41e-05 |
| always GEMMA4_31B | 3.0968 | $0.0001338 |
| always GLM_4_6V | 2.9355 | $0.0022936 |
| always LLAMA | 3.0323 | $0.0015889 |
| always MINISTRAL_3B | 2.2581 | $0.0004114 |
| always MISTRAL | 2.2903 | $0.0003979 |
| always NEMOTRON_VL | 3.2258 | $0.0012466 |
| always OPENAI_FRONTIER_VISION | 3.5484 | $0.0045117 |
| always QWEN | 3.1613 | $0.0013033 |
| always QWEN25_VL_72B | 2.6129 | $0.0047687 |
| always QWEN3_VL_30B | 2.4516 | $0.0008602 |
| always QWEN3_VL_8B | 2.7419 | $0.0012525 |

Overlay written to `config/routing.tuned.yaml` (not applied automatically).
