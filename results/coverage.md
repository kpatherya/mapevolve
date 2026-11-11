# Benchmark coverage (53 cases)

Datasets: {'demo_world': 41, 'gatech_pilot': 12} · splits: {'dev': 31, 'test': 22} · answerability: {'answerable': 45, 'unanswerable': 8} · hard negatives: 3

Label review: {'unreviewed:draft_by_claude': 41, 'reviewed:human_ground_truth_file': 6, 'reviewed:human': 6}

Cells show dev/test counts; ⚠ marks cells below 3 in either split.

| Query type | easy | moderate | hard |
| --- | --- | --- | --- |
| landmark_identification | 1/2 ⚠ | 2/0 ⚠ | 2/2 ⚠ |
| cross_view_matching | 0/2 ⚠ | 3/2 ⚠ | 4/3 |
| temporal_comparison | 3/1 ⚠ | 4/4 | 4/2 ⚠ |
| spatial_reasoning | 0/1 ⚠ | 2/0 ⚠ | 2/1 ⚠ |
| factual_retrieval | 2/0 ⚠ | 1/0 ⚠ | 1/2 ⚠ |

Expected actions: {'answer': 31, 'verify_then_answer': 13, 'decline': 2, 'use_tools': 7}
