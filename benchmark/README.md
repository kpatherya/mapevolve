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

Outputs are under `benchmark/artifacts/` and `benchmark/runs/`.
