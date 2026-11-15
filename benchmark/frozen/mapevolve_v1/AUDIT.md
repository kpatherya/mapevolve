# Pre-benchmark audit — MapEvolve v1

Audited November 15, 2026. **Suitable for an exploratory comparison**, with
AI-drafted labels still awaiting human review. No paid inference calls were
made. The `.env` file was checked for key presence and duplicate assignments;
secrets are not included in this snapshot. Authentication, balances and
successful model/image inference have not yet been smoke-tested.

## Frozen comparison settings

| Setting | Value |
| --- | --- |
| Cases | 100; 50 landmark identification, 50 factual retrieval |
| Strategies | Cheapest eligible, Jev adaptive (balanced), strongest |
| Model pool | 15 listed vision models with prior evaluation scores |
| Cheapest baseline | `GEMMA3_4B` (`google/gemma-3-4b-it`) |
| Strongest baseline | `OPENAI_FRONTIER_VISION` (`gpt-5.1`, direct OpenAI) |
| Strongest definition | Highest **prior general score**, not known strongest on these questions |
| Temperature | 0; exact deterministic reproduction is not guaranteed by hosted models |
| Output budget | 1024 tokens by default; existing 4096/8192 reasoning-model overrides retained |
| Visible answer format | JSON, 2–5 sentences; same prompt for all strategies per case |
| Images | Neutral JPEG names; no EXIF; maximum edge 2048 pixels |
| Jev | `jev-latest`, image-aware; 20-second timeout |
| Adaptive estimated budget | $0.05 per query; not a provider billing cap |
| Cascade / learned predictor / Pareto pruning | Disabled explicitly |
| Cross-model failover | Disabled; report failed selected model rather than substitute |
| Same-model retries | One SDK retry; include failures/retries in cost and latency accounting |
| Repeats | One first-pass answer per distinct case/model; identical selections reuse it |
| Scoring | Frozen 0–4 rubric; primary strict accuracy = fraction scoring 4; also score >=3 and mean |

The higher token limits for reasoning models are preserved to avoid empty
answers caused by hidden reasoning consuming the output budget. These are
**different compute budgets**, so report the overrides and actual tokens/cost.
Likewise, output truncation must be recorded, not graded as an ordinary complete
answer without noting it. Classifier and answer-check costs/time must be reported
separately from VLM-only measurements. The baseline cost estimators use historical
token measurements, including two-image pilot observations, so estimates are
approximate for this one-image benchmark.

## Checks and corrections

- Checked the 30 building-record references against the nine cited official
  Georgia Tech/athletics records in [sources.json](sources.json). For example,
  the Kendeda record distinguishes [completion, classes and certification](https://livingbuilding.gatech.edu/about),
  and the tower record distinguishes [George and Scheller programs and heights](https://news.gatech.edu/news/2026/01/07/tech-square-3-reaches-major-milestone).
- Checked all 12 metadata answers against original `DateTimeOriginal` fields.
  Those fields are evidence of a recorded date, not independent authentication.
- Schema, counts, case IDs, image paths and CSV alignment passed validation.
  All labels remain `reviewed=false`; no claim of human validation is made.
- Checked the current [OpenRouter public model catalog](https://openrouter.ai/api/v1/models).
  The 15 active model listings support images. Claude Sonnet 4 is absent and
  remains excluded. Listing does not prove account access or image API success.
- Updated Nemotron output pricing in this snapshot from $0.47 to $0.50 per
  million output tokens. The mutable default registry was left unchanged.
- Replaced stale registry limitations claiming only six temporal cases existed:
  current scores include the prior 53-case round and task-specific metrics.
- Disabled the learned predictor by setting its path to null, so a future
  training run cannot silently activate it during this benchmark.
- Clarified the frozen system prompt to permit landmark identification from
  distinctive architecture/context while retaining restrictions on unsupported
  dates/history. The earlier wording about names being visible was ambiguous
  for recognition tasks. This same prompt must be used for all strategies.
- Saved the registry, cases, answer key, source cards, prompts, full routing
  configuration, environment package versions and SHA-256 integrity manifest.
  No `MEI-*` case IDs occur in prior graded response files.

## Material limitations

**34 of the 100 cases use original images whose normalized 128-pixel thumbnail
pixels exactly match images in earlier dataset manifests.** See
[prior_image_overlap.json](prior_image_overlap.json) for case-level matches.
This is evidence of reuse, not merely the same campus location. The check is
conservative: a nonmatch does not rule out crops, recompression, or similar
views. Report full-set results and the 66-case overlap-not-detected subset
separately. Neither is a clean location-independent holdout: several buildings
already occurred in earlier routing evaluation/calibration data. The image
match report is against earlier manifests, not proof that every matched case
contributed to the model's prior score.

The 100 questions reuse 51 photographs and cover 11 named landmarks. Treat
query results as correlated and include averages by location. Hard interior,
blurred and close-up identity labels remain provisional. A campus expert's
review is needed before describing the answer key as verified ground truth.

Factual retrieval is closed-world: 30 supplied-record tasks, 12 supplied-EXIF
tasks, and 8 missing-evidence controls. Report these separately. This does not
measure autonomous web retrieval or unaided knowledge of building history.

Model names are mostly hosted aliases; OpenRouter may select different
providers. Capture actual returned model IDs/provider names. The catalog's
OpenAI listing is only a pricing/modality reference; direct OpenAI account
access was not authenticated. `jev-latest` may also change externally.

The prior model ranking was largely judged by GPT-5.1, including its own
answers. Choose and record the grader model before scoring. Blind the model
identity and human-adjudicate ambiguous/disputed answers; if GPT-5.1 is used
again, disclose this self-judge bias.

## How to use this freeze

Verify without inference:

```bash
.venv/bin/python -m scripts.verify_mapevolve_freeze
```

The next batch comparison runner should call
`scripts.verify_mapevolve_freeze.load_frozen()`; it returns the checked config,
registry, cases and protocol. It **does not read environment routing overlays**.
The normal app, live-collection CLI and offline evaluator do not automatically
load this snapshot. Merely running the normal comparison checkbox will not
enforce these frozen inputs.

In particular, **do not use `evaluate_offline` unchanged for this frozen test**:
it refits registry scores and the quality predictor from the collected grades,
excluding only the current case. That answers a different question from
measuring this pre-existing router. Training scripts also do not generally
exclude test-split grades; do not train/rebuild during this run.

The future runner must also honor `protocol.json`: run both baselines even
when adaptive declines, record Jev fallback explicitly, preserve selected-model
failures, save full classification/checker distributions with case IDs, grade
all strategy answers equally, and keep test labels out of routing inputs.
These runner changes belong to the execution step; they are not implemented
by the integrity-checking loader.

The dry-run of the existing collector estimated **1,500 calls / $3.9155** for
all 15 models on all 100 cases. That is not the three-strategy comparison plan,
excludes Jev/checker/judge overhead, and would stop early with its default $1
cap. Set a deliberate total run budget when implementing the comparison batch;
the $0.05 routing preference is a separate per-query estimate.

The snapshot is reproducible against local edits via `lock.json`. If cases,
references, source code, prompts or settings change, create a new version and
rerun the audit instead of silently overwriting this freeze.
