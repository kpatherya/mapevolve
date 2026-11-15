# Benchmark status and resumption

The benchmark is incomplete because API accounts have no purchased credits.
Direct OpenAI rejected the first grading call with no credits remaining.
A separate run uses the same GPT-5.1 model through OpenRouter; OpenRouter
then reported zero total account credits. Its $20 key allowance is not an
account balance. GPT-5.1 also has a 20-request/minute new-account limit.

Saved progress:
- All 100 Jev classifications and adaptive routing decisions.
- All 100 cheapest-model answers.
- 27 adaptive strategy answer rows and 12 strongest rows (4 strongest API errors).
- 41 strategy-level grades (38 unique grading calls; some answers are shared).
- Complete exposed SDK requests/replies, errors and token/cost metadata.

Do not treat the partial accuracy table as a completed 100-case comparison.
Missing answers/grades reflect interrupted collection, not measured accuracy.

After adding OpenRouter API credits or replacing OPENROUTER_API_KEY in .env,
resume from the repository root:

```bash
.venv/bin/python -m scripts.run_frozen_benchmark \
  --run-id mapevolve-v1-openrouter-20261115 \
  --openai-via-openrouter --workers 2 \
  --resume-with-updated-runner --retry-errors
```

The updated runner paces GPT-5.1 to 15 calls/minute. It records the transport
amendment, reuses successful replies and classifications, and archives saved
API-error results before retrying them. Successful answers are not regenerated.
The original runner.py remains in this directory; resume_runner.py is the
new version. No frozen prompts, prior scores or routing settings were changed
in the transport fix. The explicit provider amendment is in run.json and
effective_registry.json. Original direct-OpenAI smoke outputs are in the
sibling mapevolve-v1-20261115 directory.

After completion, regenerate diagnostic exports:

```bash
.venv/bin/python -m scripts.analyze_frozen_benchmark \
  runs/benchmarks/mapevolve-v1-openrouter-20261115
```

Useful artifacts:
- comparison.csv / comparison.jsonl: strategy answers, grades and costs.
- full_diagnostics.csv: all exposed probabilities, checker signals, prompts'
  input evidence, reference fields, routing candidates, and parsed outputs.
- classifications.csv / classifications.jsonl: all 100 classifier results.
- routing.jsonl: action decisions and routing results for every case.
- all_sdk_usage.csv: unique exposed SDK calls, timings, tokens and identifiers.
- summary.csv / summary.json: prompt-type, difficulty, location, evidence-track,
  Jev-only and prior-overlap-subset breakdowns.
- metadata_crosschecks.csv: literal capture-date checks against the references.
- adjudication_queue.json: partial/unsupported answers for later review.
- cases/: full requests (with encoded images), raw SDK replies, errors, checks,
  judge replies/rationale and atomic checkpoints, indexed by case/model.
- frozen_inputs/: frozen case/answer-key/config/prompt snapshots.

TypeSafe classifier/checker usage is saved, but dollar cost is not returned.
Known VLM and judge costs are in cost_ledger.json. Account-level usage may also
include the first smoke test and other requests, and is not interchangeable
with the run ledger. Secrets are excluded/redacted from artifacts. Internal
provider reasoning and SDK-hidden retries are unavailable from these APIs.
