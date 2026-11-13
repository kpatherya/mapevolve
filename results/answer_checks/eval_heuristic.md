# Answer checker evaluation — heuristic (48 graded answers)

**Verdict: too weak for the cascade; keep cascade.enabled: false.** Rule of thumb: AUC >= 0.75 on unsupported claims.

| Slice | n | Unsupported | Bad (score ≤ 2) | AUC unsupported | AUC bad |
| --- | --- | --- | --- | --- | --- |
| all | 48 | 19 | 22 | 0.595 | 0.559 |
| dev | 24 | 10 | 10 | 0.682 | 0.686 |
| test | 24 | 9 | 12 | 0.478 | 0.462 |

Threshold chosen on dev (best F1 for unsupported claims): 0.2

On test at 0.2: flagged 10, precision 0.3, recall 0.333, F1 0.316.

| Threshold | Flagged | Precision | Recall | F1 (all) |
| --- | --- | --- | --- | --- |
| 0.2 | 23 | 0.435 | 0.526 | 0.476 |
| 0.3 | 15 | 0.533 | 0.421 | 0.471 |
| 0.4 | 11 | 0.545 | 0.316 | 0.4 |
| 0.5 | 5 | 0.6 | 0.158 | 0.25 |
| 0.6 | 2 | 1.0 | 0.105 | 0.19 |
| 0.7 | 1 | 1.0 | 0.053 | 0.1 |
| 0.8 | 0 | None | 0.0 | 0.0 |
