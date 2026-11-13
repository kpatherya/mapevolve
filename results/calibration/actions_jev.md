# Action threshold calibration — jev (dev only, 31 cases)

Dev action accuracy: 0.419 with the current thresholds -> **0.806** with the chosen ones.

| Setting | Current | Chosen |
| --- | --- | --- |
| tools_threshold | 0.5 | 0.55 |
| retrieval_threshold | 0.5 | 0.5 |
| verify_signal | flag | query_type |
| verify_threshold | 0.5 | 0.5 |
| decline_if_answerable_below | 0.2 | 0.2 |

Dev has no decline cases, so the decline threshold was not searched.

## Dev confusion, chosen thresholds

| expected \ chosen | answer | verify_then_answer | retrieve_then_answer | use_tools | decline |
| --- | --- | --- | --- | --- | --- |
| answer | 13 | 0 | 0 | 5 | 1 |
| verify_then_answer | 0 | 7 | 0 | 0 | 0 |
| retrieve_then_answer | 0 | 0 | 0 | 0 | 0 |
| use_tools | 0 | 0 | 0 | 5 | 0 |
| decline | 0 | 0 | 0 | 0 | 0 |

## Config block

```yaml
actions:
  by_source:
    jev:
      tools_threshold: 0.55
      retrieval_threshold: 0.5
      verify_signal: query_type
      verify_threshold: 0.5
      decline_if_answerable_below: 0.2
```
