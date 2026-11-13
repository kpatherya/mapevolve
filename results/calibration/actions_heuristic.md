# Action threshold calibration — heuristic (dev only, 31 cases)

Dev action accuracy: 0.710 with the current thresholds -> **0.968** with the chosen ones.

| Setting | Current | Chosen |
| --- | --- | --- |
| tools_threshold | 0.5 | 0.5 |
| retrieval_threshold | 0.5 | 0.5 |
| verify_signal | flag | flag |
| verify_threshold | 0.5 | 0.65 |
| decline_if_answerable_below | 0.2 | 0.2 |

Dev has no decline cases, so the decline threshold was not searched.

## Dev confusion, chosen thresholds

| expected \ chosen | answer | verify_then_answer | retrieve_then_answer | use_tools | decline |
| --- | --- | --- | --- | --- | --- |
| answer | 18 | 0 | 0 | 1 | 0 |
| verify_then_answer | 0 | 7 | 0 | 0 | 0 |
| retrieve_then_answer | 0 | 0 | 0 | 0 | 0 |
| use_tools | 0 | 0 | 0 | 5 | 0 |
| decline | 0 | 0 | 0 | 0 | 0 |

## Test (22 cases, scored once)

Test action accuracy: 0.636 current -> **0.864** chosen.

| expected \ chosen | answer | verify_then_answer | retrieve_then_answer | use_tools | decline |
| --- | --- | --- | --- | --- | --- |
| answer | 11 | 1 | 0 | 0 | 0 |
| verify_then_answer | 0 | 6 | 0 | 0 | 0 |
| retrieve_then_answer | 0 | 0 | 0 | 0 | 0 |
| use_tools | 0 | 0 | 0 | 2 | 0 |
| decline | 1 | 0 | 0 | 1 | 0 |

## Config block

```yaml
actions:
  by_source:
    heuristic_fallback:
      tools_threshold: 0.5
      retrieval_threshold: 0.5
      verify_signal: flag
      verify_threshold: 0.65
      decline_if_answerable_below: 0.2
```
