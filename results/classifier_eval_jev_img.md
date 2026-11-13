# Classifier evaluation — jev (image-aware)

Sources used: {'jev': 53} · cases: 53 · labels: {'unreviewed': 41, 'reviewed': 12}

| Slice | n | Type acc | Type macro-F1 | Type ECE | Diff acc | Diff within-1 | Action acc | Tools P/R | Verify P/R | Answerable acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 53 | 1.0 | 1.0 | 0.0272 | 0.6415 | 0.9811 | 0.8868 | 0.6364/1.0 | 0.3103/0.6923 | 0.7925 |
| dataset=demo_world | 41 | 1.0 | 1.0 | 0.0283 | 0.6829 | 0.9756 | 0.878 | 0.5556/1.0 | 0.3158/0.6 | 0.7561 |
| dataset=gatech_pilot | 12 | 1.0 | 1.0 | 0.0233 | 0.5 | 1.0 | 0.9167 | 1.0/1.0 | 0.3/1.0 | 0.9167 |
| split=dev | 31 | 1.0 | 1.0 | 0.0429 | 0.6129 | 0.9677 | 0.9032 | 0.7143/1.0 | 0.3125/0.7143 | 0.7742 |
| split=test | 22 | 1.0 | 1.0 | 0.005 | 0.6818 | 1.0 | 0.8636 | 0.5/1.0 | 0.3077/0.6667 | 0.8182 |

## Difficulty confusion (rows = label, cols = predicted)

| | easy | moderate | hard |
| --- | --- | --- | --- |
| easy | 6 | 5 | 1 |
| moderate | 2 | 10 | 6 |
| hard | 0 | 5 | 18 |

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
| answer | 27 | 1 | 0 | 2 | 1 |
| verify_then_answer | 0 | 13 | 0 | 0 | 0 |
| retrieve_then_answer | 0 | 0 | 0 | 0 | 0 |
| use_tools | 0 | 0 | 0 | 7 | 0 |
| decline | 0 | 0 | 0 | 2 | 0 |
