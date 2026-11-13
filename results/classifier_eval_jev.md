# Classifier evaluation — jev

Sources used: {'jev': 53} · cases: 53 · labels: {'unreviewed': 41, 'reviewed': 12}

| Slice | n | Type acc | Type macro-F1 | Type ECE | Diff acc | Diff within-1 | Action acc | Tools P/R | Verify P/R | Answerable acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 53 | 1.0 | 1.0 | 0.0164 | 0.4906 | 0.8679 | 0.4151 | 0.3684/1.0 | 0.3714/1.0 | 0.283 |
| dataset=demo_world | 41 | 1.0 | 1.0 | 0.0166 | 0.4878 | 0.8293 | 0.439 | 0.3125/1.0 | 0.4167/1.0 | 0.2927 |
| dataset=gatech_pilot | 12 | 1.0 | 1.0 | 0.0158 | 0.5 | 1.0 | 0.3333 | 0.6667/1.0 | 0.2727/1.0 | 0.25 |
| split=dev | 31 | 1.0 | 1.0 | 0.0245 | 0.4194 | 0.8387 | 0.4194 | 0.4167/1.0 | 0.35/1.0 | 0.2581 |
| split=test | 22 | 1.0 | 1.0 | 0.005 | 0.5909 | 0.9091 | 0.4091 | 0.2857/1.0 | 0.4/1.0 | 0.3182 |

## Difficulty confusion (rows = label, cols = predicted)

| | easy | moderate | hard |
| --- | --- | --- | --- |
| easy | 1 | 4 | 7 |
| moderate | 1 | 5 | 12 |
| hard | 0 | 3 | 20 |

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
| answer | 3 | 19 | 0 | 9 | 0 |
| verify_then_answer | 0 | 12 | 0 | 1 | 0 |
| retrieve_then_answer | 0 | 0 | 0 | 0 | 0 |
| use_tools | 0 | 0 | 0 | 7 | 0 |
| decline | 0 | 0 | 0 | 2 | 0 |
