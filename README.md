# mapevolve
Georgia Tech campus change exploration with geotagged imagery, multimodal retrieval, and evidence-based query planning.

## Jev-based adaptive VLM routing (prototype)

TypeSafe **Jev** classifies a visual query. Deterministic code then decides
*what to do* (answer, verify, use a tool, decline) and *which VLM* to call,
weighing task-specific quality, cost and latency. The model runs through
`ChatOpenRouter` or `ChatOpenAI`, and its answer can be checked and escalated
once.

```
question + 1–2 images (+ optional supplied evidence)
      │
      ├─ pair features (SIFT alignment, appearance similarity, archival flag)
      ▼
JevQueryClassifier ──(fails / no key)──► heuristic fallback (labelled)
  one TypeSafeClassifier.invoke, 7 typed questions:
  query_type · difficulty · visual_expertise · needs_image_retrieval ·
  needs_pair_verification · needs_external_tools · answerable_from_inputs
      ▼
decide_action      answer | verify_then_answer | retrieve_then_answer | use_tools | decline
  (coordinate calculator runs when coordinates are supplied)
      ▼
RoutingPolicy      filters → expected quality (registry table, or learned
                   predictor once trained on ≥ 30 cases) → economy | balanced | accuracy
      ▼
VLMExecutor        same system prompt + task block for Jev's query type,
                   JSON output, temperature 0, timeout, failover
      ▼
AnswerChecker      image-aware Jev check (heuristic without a key)
  └─ optional cascade: escalate once to the accuracy pick if risk ≥ 0.4
      ▼
trace (runs/routing_traces/) + 👍/👎 feedback (runs/feedback/)
```

### Data

| Manifest | Cases | Labels | Source |
| --- | --- | --- | --- |
| `datasets/gatech_pilot.json` | 12 (6 graded pairs, 4 smoke, 2 adversarial cross-view) | team-written, reviewed | `benchmark` branch artifacts |
| `datasets/demo_world.json` | 41 world landmarks, all 5 query types | **drafted by Claude, unreviewed** | 51 Wikimedia Commons images |

All manifests use one schema (`jev_router/dataset.py`). Each case records
`query_type`, `difficulty`, `answerable`, `expected_action`, a reference
answer with required points and claims to avoid, the label author, and review
status. Dev/test splits are assigned **per location**, and the validator
rejects any location or image that appears in both splits. Only the question,
images and supplied evidence are routing inputs (`routing_inputs()`).
Everything else is label data.

The demo images (39 MB, CC0 / public domain / CC BY / CC BY-SA, no NC/ND)
come from `data/demo_world/sources.yaml`. Author, license, source page and
SHA-256 for each file are in `data/demo_world/attribution.json`. One Commons
file titled "Casa Milà" actually shows Casa Batlló (its signage reads "CASA
BATLLÓ", no. 43). It is relabelled, and the error is kept as a test case.
The demo references are drafts written after viewing the images; review them
before quoting results on this set.

`python -m scripts.coverage_report` prints the query type × difficulty ×
split grid and flags thin cells (`results/coverage.md`). Most cells still hold
fewer than 3 cases per split.

### Setup

Requires Python ≥ 3.11 (`langchain-typesafe` requires it).

```bash
python3.12 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Fill in `TYPESAFE_API_KEY`, `OPENROUTER_API_KEY` and `OPENAI_API_KEY` in
`.env`; it is git-ignored, and every key is read from the environment. Each
missing key degrades one stage gracefully:

* No TypeSafe key: classification and answer checks use the labelled heuristic.
* No VLM keys: the app still classifies, picks an action and routes, but skips the answer.

### Run

Start the demo (pick a demo case or upload images):

```bash
streamlit run app.py
```

Refresh current OpenRouter prices and availability:

```bash
python -m scripts.build_registry
```

Rebuild the manifests and check coverage:

```bash
python -m scripts.build_gatech_manifest && python -m scripts.build_demo_manifest && python -m scripts.coverage_report
```

Score the classifier against the labels (type, difficulty, action, flags, calibration):

```bash
python -m scripts.evaluate_classifier
```

Train the learned predictor and compare it with the registry table (leave-one-case-out):

```bash
python -m scripts.train_quality_model
```

Compare routing strategies on recorded graded responses:

```bash
python -m scripts.evaluate_offline
```

Collect new graded data. The first command is a dry run that estimates cost; drop
`--dry-run` to call the APIs, capped by `--max-cost`:

```bash
python -m scripts.run_live_benchmark --dataset demo_world --dry-run
```

```bash
python -m scripts.grade_responses judge results/live_runs/<run>.jsonl --model gpt-5.1
```

```bash
python -m scripts.grade_responses export results/live_runs/<run>.jsonl
```

After filling `human_score` in `results/review/<run>.csv`:

```bash
python -m scripts.grade_responses import results/review/<run>.csv
```

Graded rows in `results/graded/` feed `build_registry`, `train_quality_model`,
`evaluate_offline`, `pareto_report` and `tune_router` automatically. Human grades override judge grades. A full
demo-world run on all 7 listed models is estimated at about $1.70, or $0.11
for Gemma, Mistral, Qwen and Llama.

Build the Pareto report (per-cell model frontiers plus the router's own
cost–quality curve; open `results/pareto/index.html`):

```bash
python -m scripts.pareto_report
```

Tune routing settings on the dev split. This writes a candidate overlay; it goes to
`config/routing.tuned.yaml` only once dev has ≥ 20 graded cases:

```bash
python -m scripts.tune_router
```

Score the frozen overlay on the test split, once:

```bash
python -m scripts.tune_router --evaluate-test
```

Use a tuned overlay in the app or scripts:

```bash
JEV_ROUTING_OVERLAY=config/routing.tuned.yaml streamlit run app.py
```

Run the tests:

```bash
python -m pytest
```

### Live evaluation

Step-by-step instructions for the first real run (keys, spending caps, label
review, Jev evaluation, live benchmark, judge and human grading, analysis)
are in [docs/RUNBOOK.md](docs/RUNBOOK.md). Supporting scripts:
`scripts.judge_agreement` (judge vs human κ), `scripts.evaluate_answer_check`
(can the checker spot bad answers?), `grade_responses export --sample 0.2`
(blind, stratified human sample), and `run_live_benchmark --repeats N`.

### Pareto frontiers

`scripts.pareto_report` compares models on four objectives: quality (max),
unsupported-claim rate (min), cost per query (min) and mean latency (min).
Three more axes join automatically once live runs provide the data:

* refusal quality on unanswerable cases
* reliability (ok rate, JSON compliance)
* consistency across repeated runs, plus p90 latency

Frontiers are computed for every cell: all cases, each query type, and each
type × difficulty, using only cases graded for every model. Each model also
gets a **membership** score, the share of case-level bootstrap resamples in
which it stays on the frontier, because a single frontier from a few cases is
fragile.

The report also replays the router under 50 settings (modes × budgets ×
Balanced weights, leave-one-case-out) to show where the router sits relative to
the single-model frontier.

Current pilot frontier (6 cases): **Gemma 3 27B (100% membership), Mistral Small
3.2 (60%) and Llama 4 Maverick (57%, there for its 0% unsupported-claim rate)**.
GPT-5.1, Gemini, Qwen, Claude and Kimi are each beaten on all four objectives by
a cheaper model. On these 6 cases no router setting beats always-Gemma, because
Gemma is both the cheapest and on the frontier.

With `pareto.prune_dominated: true`, the policy drops models whose membership
is below 20% in the query's cell. It uses the most specific cell with ≥ 5
graded cases and never removes the last eligible model. Pruning is off by
default until cells have enough data; the app has a toggle and shows the
frontier under "Model frontier (Pareto)".

### Tuning

`scripts.tune_router` grid-searches the Economy floor, Balanced weights,
shrinkage strength and unsupported-claim penalty. That is 200 configs, run on
the **dev split only**, with the registry rebuilt from dev grades alone, leave-
one-case-out. Ties prefer the current settings, so nothing changes without
evidence. `--evaluate-test` scores the frozen overlay on test, logs it to
`results/tuning/test_runs.jsonl`, and warns if test was already scored with a
different overlay.

Today dev has 3 graded cases, so most configs tie (e.g. all 20 accuracy
configs). The candidate overlay stays in `results/tuning/` and the shipped
defaults are unchanged. Tuning becomes meaningful after the live-benchmark and
grading loop above.

### Round 1 routing results (2026-11-14)

14 models x 53 cases (742 answers, $2.10), graded by the GPT-5.1 judge against
references that are still mostly unreviewed drafts. Two caveats apply to every
number here: the judge also grades its own model's answers (possible
self-preference; the blind human sample will test this), and 41 of 53
references are unreviewed.

**Held out (leave-one-case-out, all 53 cases, default settings):**

| Strategy | Score /4 | Cost/answer |
| --- | --- | --- |
| Always GPT-5.1 | 3.58 | $0.0045 |
| Jev Accuracy | 3.34 | $0.0042 |
| Jev Balanced | 2.94 | $0.00012 |
| Always cheapest (Gemma 3 4B) | 2.51 | $0.00005 |
| Best possible per case (oracle) | 3.87 | $0.0016 |

**Tuned on dev (31 cases), scored once on test (22 cases)**, overlay
`config/routing.tuned.yaml` (hash in `results/tuning/test_runs.jsonl`; not applied):

| Mode | Dev score / cost | Test score / cost | Single model that beats it on test |
| --- | --- | --- | --- |
| Economy | 3.23 / $0.00026 | 3.09 / $0.00017 | Gemma 3 27B: 3.36 / $0.00012 |
| Balanced | 3.29 / $0.0024 | 3.27 / $0.0020 | GLM-4.6V: 3.64 / $0.0021 |
| Accuracy | 3.45 / $0.0039 | 3.45 / $0.0034 | GLM-4.6V: 3.64 / $0.0021 |

**Reading.** On test, every tuned mode is beaten by some single model. The cause
is noise, not the routing idea: model rankings move a lot between dev and test
(GLM-4.6V 2.94 -> 3.64, Nemotron VL 3.23 -> 3.18; the dev-test correlation of
model means is 0.69), and with 2-9 cases per query-type x difficulty cell the
router chases dev-set luck. The oracle (3.87 at a third of GPT-5.1's cost) shows
the headroom is real. What would let routing capture it: many more cases per
cell (aim for 20+), repeats to measure run-to-run noise, coarser cells (query
type only) until then, and the human grading sample to check judge bias. The
test split has now been used once; the next tuning round needs fresh test cases.

Tuning fixes made before test was scored: each mode now carries its own
`shrinkage_k` (one shared value had silently replaced Economy's tuned value), and
Balanced is chosen as the cheapest config within 0.25 of its best score (the
old "highest score" rule turned it into Accuracy mode).

### Jev classifier results (live TypeSafe, 2026-11-13)

`results/classifier_eval_jev_img.md` and `results/calibration/`, 53 cases (41 labels still unreviewed):

| | Fallback | Jev (text) | Jev (image-aware) |
| --- | --- | --- | --- |
| Query-type accuracy / calibration error | 1.00 / 0.40 | 1.00 / 0.016 | 1.00 / 0.027 |
| Difficulty accuracy | 0.58 | 0.49 | **0.64** |
| AUC: answerable | 0.94 | 0.98 | **0.996** |
| AUC: needs a tool | 0.99 | 0.97 | 0.98 |
| AUC: needs pair verification | 0.99* | 0.52 | 0.54 |
| Action accuracy, dev, calibrated thresholds | 0.97* | 0.81 | **0.90** |
| Action accuracy, test, scored once | 0.86* | not scored | **0.86** |

\* Partly circular: the fallback's keyword rules and the drafted labels have the same author.

Jev's own pair-verification flag is at chance, so for Jev the verify decision
uses P(query type = cross-view matching), which Jev gets right and well
calibrated (`actions.by_source.jev` in `config/routing.yaml`, chosen on dev by
`python -m scripts.calibrate_actions --classifier jev --image-aware`). Image-aware
Jev is now the default (`jev.image_aware: true`); TypeSafe needs the opt-in
header `TypeSafe-Danger-Zone-Images: true`, which the client sends. Decline
thresholds are uncalibrated: the only two decline cases are in test, and both are missed.

### Current results (heuristic fallback, before live keys)

Everything below uses the **heuristic fallback**, not Jev, because no
TypeSafe key was set. The numbers measure the harnesses as much as the router.

**Classifier vs. labels** (`results/classifier_eval_heuristic.md`, 53 cases):

| Metric | Value | Note |
| --- | --- | --- |
| Query-type accuracy | 1.00 | Inflated: I wrote both the demo questions and the keyword rules |
| Difficulty accuracy | 0.58 (within one level: 0.94) | The real gap; image-aware Jev is the intended fix |
| Action accuracy | 0.68 | Most misses: unaligned temporal pairs get `verify_then_answer` instead of `answer` (adds a cautionary note) |
| Answerable-from-inputs accuracy | 0.98 | |

The harness found a real bug on its first run: 12 of 14 cross-view questions
were typed as spatial because they mention "distance". That bug is fixed.

**Learned predictor vs. registry table** (`results/quality_model_eval.md`, leave-one-case-out, 48 graded responses):

| Estimator | MAE (score/4) | Top-1 regret |
| --- | --- | --- |
| Per-model general mean | 0.344 | 0.67 |
| Registry shrunk table (current policy) | 0.336 | 0.67 |
| Learned predictor | 0.340 | 0.67 |

There is no gain at 6 cases, so the policy uses the predictor only once it has
≥ 30 graded cases (`quality_model.min_cases`).

**Routing strategies** (`results/offline_eval_summary.csv`, leave-one-case-out, 6 graded pilot cases):

| Strategy | Mean score /4 | Unsupported | Avg cost | Regret |
| --- | --- | --- | --- | --- |
| Always cheapest (Gemma) | 3.00 | 1/6 | $0.00009 | 0.33 |
| Always strongest | 2.67 | 3/6 | $0.00073 | 0.67 |
| Fixed task-based | 2.50 | 3/6 | $0.0045 | 0.83 |
| Jev adaptive – economy / balanced | 3.00 | 1/6 | $0.00016 / $0.00009 | 0.33 |
| Jev adaptive – accuracy (table or learned) | 2.67 | 3/6 | $0.00073 | 0.67 |
| Jev cascade (balanced + check + escalate) | 2.83 | 2/6 | $0.00064 | 0.50 |
| Oracle | 3.33 | 0/6 | $0.00016 | 0 |

**The cascade is off by default.** The heuristic answer checker reaches only
AUC 0.60 against the judge's unsupported-claim labels. Both escalations were
false alarms: Gemma's answers there scored 4 and 3, and the "accuracy" picks
scored 4 and 2. Without leave-one-out, "always strongest" scores 3.17
(`results/in_sample/`), because the best single model changes when one case is
held out.

### Provisional choices, tuned on the data they are reported on

These parts have no independent validation yet, so re-check them on new data:

* monochrome threshold (chroma spread < 18) and SIFT alignment thresholds (≥ 20 inliers and inlier ratio ≥ 0.4 means "aligned")
* the fallback's keyword rules, including the cross-view-before-spatial order
* cascade risk threshold 0.4. I removed a word list from the checker after
  noticing it came from the same flagged answers it was scored on.

### Model registry

`config/model_registry.json` is generated from `config/models.yaml`, the live
OpenRouter models API, and the pilot results:

| Slot | Model ID (as evaluated) | Backend | Mean score /4 | Unsupported rate | OpenRouter status (2026-11-12) |
| --- | --- | --- | --- | --- | --- |
| OPENAI_FRONTIER_VISION | `gpt-5.1` → `gpt-5.1-2025-11-13` | OpenAI direct | 3.00 | 0.50 | reference price from `openai/gpt-5.1` |
| GEMINI | `google/gemini-3.1-pro-preview` | OpenRouter | 2.83 | 0.50 | listed |
| CLAUDE | `anthropic/claude-sonnet-4` | OpenRouter | 2.67 | 0.67 | **not listed**, excluded from live routing |
| QWEN | `qwen/qwen3-vl-235b-a22b-instruct` | OpenRouter | 3.00 | 0.33 | listed |
| GEMMA | `google/gemma-3-27b-it` | OpenRouter | 3.00 | 0.17 | listed |
| LLAMA | `meta-llama/llama-4-maverick` | OpenRouter | 2.67 | 0.00 | listed |
| MISTRAL | `mistralai/mistral-small-3.2-24b-instruct` | OpenRouter | 3.17 | 0.17 | listed |
| KIMI | `moonshotai/kimi-k2.5` (substituted for `kimi-vl-a3b-thinking`) | OpenRouter | 1.83 | 0.83 | listed (an expiry date of 2026-11-18 was dropped on 2026-11-12, and the price fell to $0.50/$2.50 per M tokens) |

These scores come from the GPT-5.1 LLM judge on 6 temporal-comparison cases
(n = 2 per difficulty). The human columns of `manual_review.csv` are empty;
once filled, they override the judge scores.

**Why routing currently collapses.** Every graded answer is a temporal
comparison, so all query types fall back to the same overall ranking: on all
53 cases Balanced picks Gemma and Accuracy picks Mistral. The router can only
adapt where per-cell quality differs; the round 1 benchmark supplies that.

**Round 1 pool (added 2026-11-13, not yet graded, so excluded from routing).**
Chosen from the live OpenRouter catalog to span size, cost and reasoning; all
passed a two-image smoke test with valid structured output:

| Slot | Model ID | Tier | Price in/out per M |
| --- | --- | --- | --- |
| GEMMA3_4B | `google/gemma-3-4b-it` | tiny | $0.05 / $0.10 |
| MINISTRAL_3B | `mistralai/ministral-3b-2512` | tiny | $0.10 / $0.10 |
| QWEN3_VL_8B | `qwen/qwen3-vl-8b-instruct` | small | $0.25 / $0.75 |
| QWEN3_VL_30B | `qwen/qwen3-vl-30b-a3b-instruct` | mid | $0.15 / $0.60 |
| GEMMA4_31B | `google/gemma-4-31b-it` | mid | $0.20 / $0.40 |
| NEMOTRON_VL | `nvidia/nemotron-3.5-super-vl` | reasoning (licence unverified) | $0.15 / $0.47 |
| GLM_4_6V | `z-ai/glm-4.6v` | reasoning | $0.30 / $0.90 |
| QWEN25_VL_72B | `qwen/qwen2.5-vl-72b-instruct` | large | $0.80 / $1.00 |

Tried and left out: `moondream/moondream3.1-9b-a2b` (returned the prompt's
placeholder instead of an answer) and `qwen/qwen3-vl-30b-a3b-thinking` (retired
by OpenRouter on 2026-11-15). Smoke-test hint of variety: on the Brandenburg
Gate pair, the 4B Gemma correctly said nothing changed, while GLM-4.6V and
Qwen3-VL 30B called a framing difference a physical change.

### Layout

| Path | Purpose |
| --- | --- |
| `config/models.yaml`, `config/routing.yaml` | Static model config; policy, actions, cascade and predictor settings |
| `config/prompts/` | `vlm_system_v2.txt` (JSON output), `tasks/<query_type>.txt`, `judge_v2.txt`; `vlm_system_v1.txt` kept for benchmark parity |
| `jev_router/classifier.py` | Jev questions, heuristic fallback, leakage guard |
| `jev_router/pair_features.py` | SIFT alignment, similarity, brightness and archival features |
| `jev_router/actions.py` | Action choice and coordinate calculator |
| `jev_router/policy.py`, `registry.py`, `quality_model.py` | Model selection, registry, learned predictor |
| `jev_router/vlm.py`, `answer_check.py`, `pipeline.py` | Execution, answer check, cascade, traces |
| `jev_router/dataset.py`, `graded.py`, `classify_cache.py` | Manifest schema, graded-response store, cached classifications |
| `jev_router/pareto.py`, `offline.py` | Pareto frontiers with bootstrap membership; leave-one-case-out replay of routing settings |
| `scripts/` | Registry, manifests, coverage, evaluations, Pareto report, tuning, live runs, grading, Wikimedia fetcher |
| `results/` | Committed evaluation outputs |

### Not implemented yet (by design)

Image retrieval, full pair verification, and record or search tools are not
built. The router already chooses `retrieve_then_answer`, `verify_then_answer`
and `use_tools`. It marks them `implemented: false` and tells the VLM to state
the limitation, so each component can be added to `jev_router/actions.py`
without changing the policy. The SIFT check is only an alignment signal, not
verification.

To tune for Georgia Tech, add cases to a manifest and grade live runs. Then
retrain with `train_quality_model`; once there are ≥ 30 graded cases, the
policy starts using the predictor without code changes.

## Smoke benchmark

The dependency-free smoke runner uses four representative cases from
[`benchmark/smoke_cases.json`](benchmark/smoke_cases.json). Validate the image
manifest and request payloads without making network calls:

```bash
OPENROUTER_MODELS=provider/model-a,provider/model-b \
  python3 benchmark/smoke_runner.py --dry-run
```

For live requests, export `OPENROUTER_API_KEY` in the shell where the runner is
started. Use exact OpenRouter model IDs; the runner does not silently substitute
models:

```bash
source .keys
export OPENROUTER_MODELS=provider/model-a,provider/model-b
python3 benchmark/smoke_runner.py
```

Responses and API errors are written under `runs/`, which is ignored by Git.
The runner stores raw model responses and request metadata, but never writes the
API key.

The completion budget includes hidden/provider reasoning tokens. The default is
`--max-tokens 1200`; increase it for models that return
`finish_reason: "length"`, or request less provider reasoning with
`--reasoning-effort low`. A truncated response is recorded as
`status: "truncated_output"` rather than being treated as a valid answer.
