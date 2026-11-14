# Live evaluation runbook

How to go from the pilot (6 graded cases, heuristic classifier) to real Jev and
VLM measurements. Steps are in order: each one depends on the one before.
Estimated total: about $10 in VLM and judge calls, plus TypeSafe usage.

Run every command from the repo root with the virtual environment active.

## 1. Keys and spending caps (account owner)

```bash
cp .env.example .env
```

Fill in `TYPESAFE_API_KEY`, `OPENROUTER_API_KEY` and `OPENAI_API_KEY` in `.env`.
The file is git-ignored; never paste keys into chat, issues or commits.

* **OpenRouter:** add credits and set a per-key credit limit (e.g. $15). The
  original pilot stopped mid-run on a key limit ("402: requires more credits").
* **OpenAI:** set a monthly usage limit (e.g. $10).
* **TypeSafe:** check plan pricing. This runbook makes about 110 classification
  calls (step 4) and about 400 answer checks (step 6).

## 2. Smoke test (≈ $0.01)

```bash
python -m scripts.build_registry
```

```bash
python -m scripts.run_live_benchmark --cases W-T-bt-same-day --models GEMMA --run-id smoke --max-cost 0.05
```

Then open the app and run one demo case with "Call the VLM" ticked. Check that
the classification says `source: jev`, not `heuristic_fallback`:

```bash
streamlit run app.py
```

## 3. Review the demo labels (human, 1–2 hours) — before any grading

Every grade is judged against these references, so this comes first. In the app,
pick each demo case and expand "Reference answer". Fix anything wrong in
`data/demo_world/cases.yaml`, then add to each checked case:

```yaml
    reviewed: true
    reviewer: <your name>
    review_note: <optional: what you changed>
```

Rebuild and check progress (the coverage report counts reviewed cases):

```bash
python -m scripts.build_demo_manifest && python -m scripts.coverage_report
```

`T-fk-rebuilt` carries an explicit "reviewer: verify" note. Grading warns about
unreviewed cases, and `--reviewed-only` skips them.

## 4. Is Jev better than the fallback? (TypeSafe calls only)

```bash
python -m scripts.evaluate_classifier --classifier jev
```

```bash
python -m scripts.evaluate_classifier --classifier jev --image-aware
```

Compare `results/classifier_eval_jev*.md` with `classifier_eval_heuristic.md`.
The script warns if Jev fell back on any case; such a report is partly heuristic.

Then choose action thresholds for Jev on dev only (no API calls; uses the cache),
put the printed block under `actions.by_source` in `config/routing.yaml`, and
score test once with `--evaluate-test`:

```bash
python -m scripts.calibrate_actions --classifier jev --image-aware
```

Done on 2026-11-13: image-aware Jev won (dev action accuracy 0.90, test 0.86) and
is now the default; see the README "Jev classifier results". Re-run both steps
after the label review changes any labels.

## 5. Live benchmark (≈ $2.50)

Model notes from the 2026-11-13 smoke test: Kimi K2.5 returns empty answers (it
reasons until the output limit, even with reasoning off), so leave it out with
`--models` or fix it first; Claude Sonnet 4 is no longer on OpenRouter and is
skipped until a replacement is chosen in `config/models.yaml`.

Full pass over all 53 cases with every listed model, including the round 1
pool (14 models without Kimi: est. $1.85; with Kimi $2.42):

```bash
python -m scripts.run_live_benchmark --run-id round1 --max-cost 5
```

To leave Kimi out, list the models explicitly with `--models` (see
`config/models.yaml`; `--dry-run` prints the estimate).

Two more passes on the cheap models, for consistency and latency spread (est. $0.28):

```bash
python -m scripts.run_live_benchmark --run-id round1-cheap --models GEMMA MISTRAL QWEN LLAMA --repeats 2 --max-cost 1
```

Runs resume where they stopped; re-run the same command after any interruption.

## 6. Grading (≈ $2 judge; 2–3 hours human)

AI judge on everything:

```bash
python -m scripts.grade_responses judge results/live_runs/round1.jsonl --model gpt-5.1
```

```bash
python -m scripts.grade_responses judge results/live_runs/round1-cheap.jsonl --model gpt-5.1
```

Blind human sample, about 20% of answers spread across query type, difficulty
and model. Graders see the images, question, reference and answer, but not the
judge's score:

```bash
python -m scripts.grade_responses export results/live_runs/round1.jsonl --sample 0.2
```

Fill in `human_score` (0–4), `human_label`, `human_unsupported_claim` and
`grader` in `results/review/round1.sample20.csv`, then:

```bash
python -m scripts.grade_responses import results/review/round1.sample20.csv
```

How far can the judge be trusted?

```bash
python -m scripts.judge_agreement
```

If weighted kappa is below 0.6 or the unsupported-flag kappa is below 0.5, grade
more by hand (human grades override judge grades automatically).

Does the image-aware Jev answer check spot bad answers?

```bash
python -m scripts.evaluate_answer_check
```

The heuristic checker scores AUC 0.60 overall and 0.48 on test (chance). Turn
on `cascade.enabled` only if Jev clears about 0.75, using the threshold the
script picks on dev.

## 7. Analysis and tuning (free)

```bash
python -m scripts.build_registry && python -m scripts.coverage_report && python -m scripts.pareto_report
```

```bash
python -m scripts.train_quality_model && python -m scripts.evaluate_offline && python -m scripts.tune_router
```

Decide, from the outputs, whether to turn on Pareto pruning
(`pareto.prune_dominated`), the learned predictor (automatic at ≥ 30 graded
cases) and the cascade. Then score the frozen settings on test, **once**:

```bash
python -m scripts.tune_router --evaluate-test
```

## 8. Report

Update the README results and the colleague guide with the real numbers, then
commit. Next round: add Georgia Tech cases where `results/coverage.md` shows
thin cells (aim for about 5 per cell per split) and repeat steps 3–7.
