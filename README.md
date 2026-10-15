# mapevolve
Georgia Tech campus change exploration with geotagged imagery, multimodal retrieval, and evidence-based query planning.

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
