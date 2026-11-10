# MapEvolve VLM benchmark

Six same-location pairs. The model is given both images and asked what changed. No retrieval.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r benchmark/requirements.txt
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
```

Optional: `OPENAI_VLM_MODEL`, `OPENAI_JUDGE_MODEL`.

Put the image zip in this directory as `IMAGE PAIRS.zip` or `IMAGE PAIRS 2.zip`.

## Run

```bash
python -m benchmark.src.prepare_data
python -m benchmark.src.validate_data
python -m benchmark.src.run_benchmark --dry-run
python -m benchmark.src.run_benchmark --cases E1 --smoke-test
python -m benchmark.src.run_benchmark --resume
python -m benchmark.src.judge_results --resume
python -m benchmark.src.aggregate_results
python -m benchmark.src.make_reports
python -m benchmark.src.audit
python -m pytest benchmark/tests
```

A finished pair is skipped on the next run. `--force` runs it again.

## Hard same-building queries

Run the separate interior/exterior and adversarial landmark query set with:

```bash
python -m benchmark.src.run_building_queries
```

The runner uses the resolved models in `benchmark/artifacts/resolved_models.json`
when available and writes all answers, per-model errors, latency, token usage, and
the reference case metadata to
`benchmark/runs/building_query_results.json`. Use `--dry-run` to write the
case/model manifest without making API calls, or `--models GEMINI CLAUDE` to
run a subset.

Outputs are under `benchmark/artifacts/` and `benchmark/runs/`.
