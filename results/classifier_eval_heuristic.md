# Classifier evaluation — heuristic

Sources used: {'heuristic_fallback': 53} · cases: 53 · labels: {'unreviewed': 41, 'reviewed': 12}

| Slice | n | Type acc | Type macro-F1 | Type ECE | Diff acc | Diff within-1 | Action acc | Tools P/R | Verify P/R | Answerable acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 53 | 1.0 | 1.0 | 0.4 | 0.5849 | 0.9434 | 0.6792 | 0.7778/1.0 | 0.4815/1.0 | 0.9811 |
| dataset=demo_world | 41 | 1.0 | 1.0 | 0.4 | 0.561 | 0.9268 | 0.6829 | 0.7143/1.0 | 0.5/1.0 | 0.9756 |
| dataset=gatech_pilot | 12 | 1.0 | 1.0 | 0.4 | 0.6667 | 1.0 | 0.6667 | 1.0/1.0 | 0.4286/1.0 | 1.0 |
| split=dev | 31 | 1.0 | 1.0 | 0.4 | 0.6129 | 0.9355 | 0.7097 | 0.8333/1.0 | 0.4667/1.0 | 1.0 |
| split=test | 22 | 1.0 | 1.0 | 0.4 | 0.5455 | 0.9545 | 0.6364 | 0.6667/1.0 | 0.5/1.0 | 0.9545 |

## Difficulty confusion (rows = label, cols = predicted)

| | easy | moderate | hard |
| --- | --- | --- | --- |
| easy | 5 | 4 | 3 |
| moderate | 3 | 7 | 8 |
| hard | 0 | 4 | 19 |

## Query-type confusion

| | landmark_ide | cross_view_m | temporal_com | spatial_reas | factual_retr |
| --- | --- | --- | --- | --- | --- |
| landmark_identification | 9 | 0 | 0 | 0 | 0 |
| cross_view_matching | 0 | 14 | 0 | 0 | 0 |
| temporal_comparison | 0 | 0 | 18 | 0 | 0 |
| spatial_reasoning | 0 | 0 | 0 | 6 | 0 |
| factual_retrieval | 0 | 0 | 0 | 0 | 6 |

## Action confusion (rows = expected, cols = chosen)

| | answer | verify_then_answer | retrieve_then_answer | use_tools | decline |
| --- | --- | --- | --- | --- | --- |
| answer | 16 | 14 | 0 | 1 | 0 |
| verify_then_answer | 0 | 13 | 0 | 0 | 0 |
| retrieve_then_answer | 0 | 0 | 0 | 0 | 0 |
| use_tools | 0 | 0 | 0 | 7 | 0 |
| decline | 1 | 0 | 0 | 1 | 0 |
