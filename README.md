# MapEvolve

MapEvolve explores Georgia Tech campus changes with geotagged imagery,
multimodal retrieval, and evidence-based query planning. The repository
currently contains a dependency-free MVP vertical slice for routing
same-building verification requests to an OpenRouter vision model, plus a
four-case image smoke benchmark.

## What is implemented

The current MVP runs the following flow:

```text
validate request -> classify query -> select model -> execute backend -> write trace
```

- **Validated contracts** in `mapevolve/contracts.py` define routing requests,
  interpretations, decisions, and backend results. Inputs and outputs are
  JSON-serializable and invalid values raise visible validation errors.
- **Rule-based classification** in `mapevolve/query_classifier.py` recognizes
  `same_building_verification`, `change_captioning`,
  `evidence_backed_investigation`, `proximity_computation`, and `unknown`.
- **Evidence gating** executes only same-building requests with exactly two
  existing image files. Missing evidence produces `request_evidence`; known
  unsupported families and unknown queries produce `abstain`.
- **Static Pareto routing** in `mapevolve/pareto_policy.py` loads
  `mapevolve/model_profiles.json` and selects the lowest-cost model within the
  configured `0.02` accuracy margin of the best profile. The current
  same-building profiles include `openai/gpt-4.1-mini`,
  `anthropic/claude-3.5-sonnet`, and a dominated legacy profile.
- **Backends** in `mapevolve/backends.py` include a deterministic
  `FakeBackend` for tests and an injectable `OpenRouterBackend` for live
  requests. The live adapter sends both images, validates the structured
  `true`/`false`/`unknown` response, measures latency, and estimates cost.
- **Pipeline and CLI** in `mapevolve/pipeline.py` and `mapevolve/cli.py`
  support fake execution, OpenRouter execution, dry runs, profile selection,
  and structured JSON traces.
- **Tests** in `tests/` cover contracts, classification, Pareto selection,
  backend behavior, and end-to-end CLI/pipeline behavior using only the
  Python standard library.

The original four-case benchmark remains available in
[`benchmark/smoke_runner.py`](benchmark/smoke_runner.py). It sends the cases
from [`benchmark/smoke_cases.json`](benchmark/smoke_cases.json) directly to
OpenRouter and records provider responses. It is useful for exploratory
model comparisons, while `python -m mapevolve.cli` exercises the new routing
pipeline.

## Setup

Runtime and tests use Python's standard library; no package installation is
required. Run commands from the repository root.

Create a local `.keys` file (it is ignored by Git) containing an exported
OpenRouter key:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Load it into the current shell before any live run:

```bash
source .keys
```

Never commit `.keys` or paste the key into a trace, issue, or log. The
OpenRouter adapter does not write the key to its output.

## Recreate the MVP runs

### 1. Run the tests

```bash
python3 -m unittest discover -s tests -v
```

### 2. Run a deterministic fake request

This makes no network call and is the quickest end-to-end check:

```bash
python3 -m mapevolve.cli \
  --request-id fake-brittain-001 \
  --query "Are these the same building?" \
  --image data/brittain-dining-hall/2026-08-12-P1-001.jpg \
  --image data/brittain-dining-hall/2026-08-12-P1-002.jpg \
  --fake-answer true \
  --output runs/fake-brittain-001.json
```

Use `--fake-answer false` or `--fake-answer unknown` to exercise the other
valid answer states.

### 3. Build an OpenRouter payload without sending it

Dry-run mode still validates the images, classifies the query, selects a
profile, and builds the request payload. It does not require an API key and
does not make a network call:

```bash
python3 -m mapevolve.cli \
  --request-id dry-run-brittain-001 \
  --query "Are these the same building?" \
  --image data/brittain-dining-hall/2026-08-12-P1-001.jpg \
  --image data/brittain-dining-hall/2026-08-12-P1-002.jpg \
  --backend-mode openrouter \
  --dry-run \
  --output runs/dry-run-brittain-001.json
```

The default profile selects `openai/gpt-4.1-mini` for this request. Use
`--profiles-path` or `--accuracy-margin` to test another static policy
configuration.

### 4. Recreate a live OpenRouter run

Load the key in the shell, then run the same request through the OpenRouter
backend:

```bash
source .keys

python3 -m mapevolve.cli \
  --request-id live-same-building-001 \
  --query "Are these the same building?" \
  --image data/brittain-dining-hall/2026-08-12-P1-001.jpg \
  --image data/brittain-dining-hall/2026-08-12-P1-002.jpg \
  --backend-mode openrouter \
  --output runs/live-same-building-001.json
```

The CLI reads `OPENROUTER_API_KEY` from the environment. The backend is
currently restricted to the two model IDs listed in
`mapevolve/backends.py`; the policy normally chooses
`openai/gpt-4.1-mini`. A live run requires network access and incurs provider
charges. The default HTTP timeout is 90 seconds; override it with
`--timeout-seconds`.

The checked-in [`logs/live-same-building-001.json`](logs/live-same-building-001.json)
is an example trace from this request. New `runs/` output is ignored by Git.
Each trace includes the request, classifier rule, routing decision, selected
profile, backend result, latency, and estimated cost. Image bytes and API
credentials are not stored in traces.

## Recreate the four-case smoke benchmark

The exploratory benchmark uses exact OpenRouter model IDs supplied through
`OPENROUTER_MODELS`. First validate the manifest and payload construction
without network access:

```bash
OPENROUTER_MODELS=provider/model-a,provider/model-b \
  python3 benchmark/smoke_runner.py --dry-run
```

For live requests, source the key and provide the models to compare:

```bash
source .keys
export OPENROUTER_MODELS=openai/gpt-4.1-mini,anthropic/claude-3.5-sonnet
python3 benchmark/smoke_runner.py
```

The smoke runner writes one result per case under `runs/`. Its four cases
cover same-building verification, viewpoint-aware comparison, an
unsupported record-level funding question, and a proximity question without
coordinates. It validates the provider's JSON shape but does not score model
answers automatically. Increase `--max-tokens` if a provider returns
`finish_reason: "length"`; use `--reasoning-effort low` when supported by the
selected provider.

## Current limitations and next steps

The MVP is intentionally narrow. Focus next on these items, in order:

1. **Measure and improve routing profiles.** Replace placeholder
   `offline_accuracy` and cost values with reproducible evaluations on
   labeled image pairs. Record the dataset split, provider/model version,
   prompt version, and scoring procedure.
2. **Finish the executable evaluation surface.** Add a scorer and regression
   fixtures for `true`, `false`, `unknown`, malformed output, timeout, and
   cost/latency reporting. Reconcile the exploratory smoke runner with the
   routed pipeline so there is one canonical run format.
3. **Add metadata-backed evidence.** The current repository has image files but
   no verified coordinate or building metadata index. Add a validated source
   for coordinates and provenance before implementing proximity computation or
   record-level investigation.
4. **Implement the remaining query families.** Add change captioning,
   evidence-backed investigation, and proximity computation only after their
   evidence requirements and failure/abstention behavior are specified.
5. **Harden operational workflows.** Add a safe environment/configuration
   guide for team runs, provider error observability, reproducible run
   manifests, and CI coverage for all no-network paths.

`plan.md` describes the intended MVP design and task boundaries; it is
background context, not a substitute for the behavior implemented in the
package today.
