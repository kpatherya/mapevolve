# Quality predictor — LOCO evaluation (48 graded responses, 6 cases)

Classifier features: heuristic. Trained query types: ['temporal_comparison'].

| Estimator | MAE (score/4) | Top-1 mean score /4 | Top-1 regret |
| --- | --- | --- | --- |
| general_mean | 0.3438 | 2.6667 | 0.6667 |
| registry_shrunk_table | 0.3363 | 2.6667 | 0.6667 |
| learned_predictor (λ=30.0) | 0.3402 | 2.6667 | 0.6667 |

Lambda chosen on the same LOCO folds; the learned MAE is slightly optimistic.
