"""Validate a complete run, write a readable report, and package all artifacts.

No API calls. Refuses to label incomplete runs as complete.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from dotenv import dotenv_values
from jev_router.dataset import ROOT
from scripts.analyze_frozen_benchmark import export
from scripts.verify_mapevolve_freeze import load_frozen


def read(p):return json.loads(p.read_text())


def finalize(run):
    load_frozen()
    export(run)
    checks=read(run/'verification.json')
    assert checks['all_100_cases_complete'], 'Case checkpoints incomplete'
    assert checks['all_300_strategy_rows_present'], 'Expected 300 strategy rows'
    assert checks['all_strategy_rows_graded'], 'Some grades are missing'
    assert all(set(statuses) <= {'ok','declined'} for statuses in checks['answer_statuses_by_strategy'].values()), 'Failed or truncated answers remain'
    mismatches=[];checked=0;forbidden=[]
    for cp in (run/'cases').iterdir():
        requests=[]
        for f in cp.glob('answers/*/vlm_sdk/request_*.json'):
            requests.append(hashlib.sha256(json.dumps(read(f)['request'],sort_keys=True).encode()).hexdigest())
        if requests:
            checked+=1
            if len(set(requests))>1:mismatches.append(cp.name)
        for f in cp.glob('classifier_sdk/request_*.json'):
            state=read(f)['request'].get('state',{})
            if any(k in state for k in ('reference','labels','difficulty','case_id','expected_action')):forbidden.append(cp.name)
    assert not mismatches, mismatches
    assert not forbidden, forbidden
    checks.update(identical_strategy_inputs_verified_for_cases=checked,input_mismatch_cases=mismatches,forbidden_label_fields_in_classifier_states=forbidden)
    (run/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    summaries=list(csv.DictReader((run/'summary_with_api_latency.csv').open()))
    def table(subset,dimension,group=None):
        lines=['| Strategy | Cases | Strict accuracy (4/4) | Score ≥3/4 | Mean score | VLM cost | API time* |',
               '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in summaries:
            if r['subset']!=subset or r['dimension']!=dimension or (group is not None and r['group']!=group):continue
            lines.append(f'| {r["strategy"]} | {r["n"]} | {float(r["strict_accuracy_all_cases"]):.1%} | {float(r["score_ge_3_all_cases"]):.1%} | {float(r["mean_score_graded"]):.3f}/4 | ${float(r["attributed_vlm_cost_usd"]):.4f} | {float(r["mean_sdk_api_seconds_excluding_pacing"]):.2f}s |')
        return lines
    status=read(run/'status.json');meta=read(run/'run.json')
    lines=['# MapEvolve benchmark results','', 'Completed: 100 cases × 3 strategies = **300 graded strategy results**.', '',
           '**These are exploratory automated grades against AI-drafted references.** Human label review and adjudication are still pending. GPT-5.1 graded all answers, including its own; the prior rankings also used GPT-5.1 grades. The text-only grader scores the parsed answer against the frozen reference; structured evidence/confidence fields are retained as diagnostics and are not independently graded.','',
           '## Full dataset','']+table('all','overall')+['',
           '*API time is the SDK request/response elapsed time, excluding client rate-limit pacing. The original per-answer `latency_seconds` includes client waiting/setup and is retained in the raw results. Neither is a statistically controlled latency benchmark; calls span different periods/provider conditions.','',
           '## Landmark identification (50 cases)','']+table('all','query_type','landmark_identification')+['',
           '## Factual retrieval (50 cases)','']+table('all','query_type','factual_retrieval')+['',
           'The factual set consists of 30 supplied-building-record questions, 12 supplied-EXIF questions, and 8 image-only unanswerable controls. It measures evidence-grounded response quality, not autonomous web search.','',
           '## Earlier-image-overlap not detected (66 cases)','']+table('overlap_not_detected','overall')+['',
           'The remaining 34 cases reuse images from earlier dataset manifests. Even the 66-case subset shares locations with prior material and is not a location-independent holdout. Multiple questions reuse the same photos; macro-location averages are provided in `location_macro_scores.json`.','',
           '## Execution and cost','',
           f'- Jev classifications: {checks["classification_sources"]}. All were reused from saved checkpoints during the funded resumption.',
           f'- Answer checks: {checks["check_sources"]} strategy rows; shared answers reuse checks.',
           f'- Unique exposed calls: '+', '.join(f'{s}: {v["calls"]}' for s,v in checks['sdk_usage_by_stage'].items())+'.',
           f'- Known unique answer + judge cost: **${status["known_vlm_and_judge_cost_usd"]:.4f}**. TypeSafe dollar costs are not returned by the SDK and are additional/unknown; their token usage is saved.',
           '- Strategy costs attribute the shared answer cost to each strategy that selects it. Do not sum the three strategy totals to calculate actual spend; use `cost_ledger.json`.',
           '- Cheapest = Gemma 3 4B. Strongest = GPT-5.1 by the frozen prior general ranking. Adaptive = balanced routing; escalation, cross-model failover, learned predictor and Pareto pruning disabled.',
           '- After direct OpenAI reported no credits, GPT-5.1 answer/grading calls were explicitly moved to OpenRouter. This backend amendment is saved in `run.json` and `effective_registry.json`; all final strongest answers in this run use that backend.',
           '- Early billing/rate-limit failures were retained. Funded resumption reused successful answers and retried saved API failures; transport changes and runner versions are archived.',
           '- Successful answers were not regenerated or cherry-picked. Same-model strategy selections share a single result. All 100 cases passed the identical-input check; no reference/label fields were found in classifier request states.','',
           '## Saved outputs','',
           '- `comparison.csv` / `comparison.jsonl`: answers, scores, errors, costs and original client timings.',
           '- `full_diagnostics.csv`: flattened classification probabilities, answerability/tool flags, routing candidates, supplied evidence, answers, checker risks, model confidence, grades and API/client timings.',
           '- `classifications.csv` / `.jsonl`, `routing.jsonl`: every classifier and routing result.',
           '- `summary.csv` / `.json` and `summary_with_api_latency.csv`: breakdowns by type, difficulty, evidence track, location, Jev-only and image-overlap subset.',
           '- `all_sdk_usage.csv`: unique SDK call IDs, model IDs, usage, exposed costs and elapsed times.',
           '- `metadata_crosschecks.csv`, `adjudication_queue.json`: deterministic date-string cross-checks and candidate answers for later review. Literal date matching is diagnostic, not a replacement for grading.',
           '- `cases/`: complete SDK requests/replies, encoded images, raw generated text, parsed JSON, errors (including previous attempts), checks, judge rationale and checkpoints.',
           '- `frozen_inputs/`, `effective_registry.json`, runner copies and `run.json`: configuration and provenance.',
           '- `verification.json`, `output_checksums.json`: completeness, prompt-equality, label-leakage checks and file hashes.','',
           'API keys are excluded/redacted. APIs do not expose private internal reasoning or SDK-hidden retry details. No outputs were added to router training or registry ingestion directories.']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Preserve the earlier blocked-state instructions instead of silently erasing them.
    if (run/'RESUME.md').exists() and not (run/'RESUME_DURING_CREDIT_BLOCK.md').exists():
        (run/'RESUME.md').rename(run/'RESUME_DURING_CREDIT_BLOCK.md')
    if (run/'account_status.json').exists() and not (run/'account_status_before_funding.json').exists():
        (run/'account_status.json').rename(run/'account_status_before_funding.json')
    (run/'completion.json').write_text(json.dumps({'completed_at_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','strategy_results':300,'all_graded':True,'verification':'passed','label_status':'AI-drafted; human review pending'},indent=2)+'\n')
    (run/'README.md').write_text('# Completed benchmark\n\nSee [REPORT.md](REPORT.md) for results and [verification.json](verification.json) for validation.\n\nThe earlier credit-block instructions/status are retained for provenance. This run is now complete.\n')
    shutil.copyfile(__file__,run/'finalize_runner.py')
    shutil.copyfile(ROOT/'scripts/analyze_frozen_benchmark.py',run/'analyze_runner.py')
    secrets=[str(v).encode() for k,v in dotenv_values(ROOT/'.env').items() if k.endswith('API_KEY') and v]
    files=[p for p in run.rglob('*') if p.is_file()]
    assert not any(secret in p.read_bytes() for p in files for secret in secrets), 'Configured API key found; refusing to package'
    if (run/'output_checksums.json').exists() and not (run/'output_checksums_partial.json').exists():
        (run/'output_checksums.json').rename(run/'output_checksums_partial.json')
    manifest={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.rglob('*')) if p.is_file() and p.name!='output_checksums.json'}
    (run/'output_checksums.json').write_text(json.dumps({'sha256':manifest,'credential_scan':'passed'},indent=2)+'\n')
    archive=run.parent/'mapevolve-v1-20261115-complete.zip'
    smoke=run.parent/'mapevolve-v1-20261115'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for directory,label in ((run,'benchmark'),(smoke,'direct_openai_smoke')):
            for p in sorted(directory.rglob('*')):
                if p.is_file():z.write(p,str(Path(label)/p.relative_to(directory)))
        for p in sorted((ROOT/'data/mapevolve_queries/images').glob('*.jpg')):z.write(p,str(Path('benchmark_images')/p.name))
        z.writestr('IMAGE_PATHS.txt','Frozen case paths data/mapevolve_queries/images/image_###.jpg map to benchmark_images/image_###.jpg in this archive. Full original model inputs are also retained as encoded images in SDK request JSON files.\n')
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        assert bad is None,bad
    print('Completed archive:',archive,round(archive.stat().st_size/1024**2,1),'MiB')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);args=p.parse_args();finalize(args.run)
