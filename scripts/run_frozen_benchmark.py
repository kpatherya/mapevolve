"""Checkpointed three-strategy benchmark, using the audited frozen inputs.

python -m scripts.run_frozen_benchmark --run-id mapevolve-v1-20261115
Use --limit 1 for an initial smoke test; rerun without it to resume all cases.
Outputs stay under runs/benchmarks, outside registry/training ingestion paths.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from jev_router.actions import decide_action, decline_message
from jev_router.answer_check import AnswerChecker
from jev_router.classifier import JevQueryClassifier, typesafe_classifier
from jev_router.dataset import ROOT, routing_inputs
from jev_router.policy import RoutingPolicy
from jev_router.schemas import QueryClassification
from jev_router.vlm import VLMExecutor, default_chat_factory
from scripts.verify_mapevolve_freeze import FREEZE, load_frozen
from scripts.grade_responses import parse_verdict

SECRETS = []
IO_LOCK = threading.Lock()
GPT_LOCK = threading.Lock()
GPT_NEXT_CALL = 0.0


def serial(value):
    if hasattr(value, 'model_dump'):
        try: return value.model_dump(mode='json')
        except Exception: return value.model_dump()
    if isinstance(value, Path): return str(value)
    return str(value)


def encode(value):
    data = json.dumps(value, default=serial, ensure_ascii=False, indent=2)
    for key in SECRETS:
        if key: data = data.replace(key, '[REDACTED]')
    return data


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic checkpoints, so interrupted writes never count as finished calls.
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(encode(value)+'\n'); tmp.replace(path)


def read(path): return json.loads(Path(path).read_text())


def utc(): return datetime.now(timezone.utc).isoformat()


class Recorder:
    """Persist complete SDK requests/replies, including all exposed metadata.

    Credentials and HTTP authorization headers are never serialized. Retry
    internals not exposed by the SDK are not fabricated as separate calls.
    """
    def __init__(self, client, directory):
        self.client, self.directory = client, Path(directory)

    def invoke(self, request, *args, **kwargs):
        name=str(getattr(self.client,'model_name',getattr(self.client,'model','')))
        if 'gpt-5.1' in name:
            # The account's GPT-5.1 limit is 20 RPM. Leave room for SDK retries.
            global GPT_NEXT_CALL
            with GPT_LOCK:
                delay=max(0,GPT_NEXT_CALL-time.monotonic())
                if delay: time.sleep(delay)
                GPT_NEXT_CALL=time.monotonic()+4.0
        self.directory.mkdir(parents=True, exist_ok=True)
        number = len(list(self.directory.glob('request_*.json')))+1
        save(self.directory/f'request_{number}.json', {'started_at':utc(), 'request':request})
        start = time.perf_counter()
        try:
            result = self.client.invoke(request, *args, **kwargs)
            save(self.directory/f'response_{number}.json', {'finished_at':utc(), 'elapsed_seconds':time.perf_counter()-start, 'response':result})
            return result
        except Exception as exc:
            save(self.directory/f'error_{number}.json', {'finished_at':utc(), 'elapsed_seconds':time.perf_counter()-start,'exception_type':type(exc).__name__,'error':str(exc)})
            raise


def judge_prompt(case, answer):
    text = (FREEZE/'prompts/judge.txt').read_text()
    ref = case.reference
    values = {'QUESTION':case.question,'EVIDENCE':case.supplied_evidence or '(none)',
              'EXPECTED_ACTION':case.expected_action,'ANSWERABLE':str(case.answerable),
              'REFERENCE':ref['answer'],'REQUIRED':'\n'.join(ref.get('required_points',[])),
              'CONFOUNDERS':'\n'.join(ref.get('acceptable_confounders',[])),
              'AVOID':'\n'.join(ref.get('unsupported_claims_to_avoid',[])),'ANSWER':answer}
    for key,value in values.items(): text=text.replace('<<<'+key+'>>>',value)
    return text


def normalize_grade(verdict):
    score = float(verdict['score'])
    if not 0 <= score <= 4: raise ValueError('Judge score out of range')
    if verdict.get('unsupported_claim'): score=min(score,3)
    return {**verdict,'score':score}


def summarize(run, cases, protocol):
    rows=[]; ledger=[]
    for c in cases:
        cp=run/'cases'/c.case_id
        if not (cp/'strategies.json').exists(): continue
        classification=read(cp/'classification.json')['classification']
        for strategy, key in read(cp/'strategies.json').items():
            folder=cp/'answers'/(key or 'adaptive_decline')
            if not (folder/'result.json').exists(): continue
            result=read(folder/'result.json'); grade=read(folder/'grade.json') if (folder/'grade.json').exists() else {}
            check=read(folder/'check.json') if (folder/'check.json').exists() else {}
            rows.append({'case_id':c.case_id,'query_type':c.query_type,'difficulty':c.difficulty,'location_id':c.location_id,'track':c.tags[0],
                         'prior_image_overlap':c.case_id in protocol['prior_image_overlap_case_ids'],
                         'strategy':strategy,'model_key':key,'classification_source':classification['source'],
                         'jev_query_type':classification['query_type']['label'],'jev_difficulty':classification['difficulty']['label'],
                         'status':result['status'],'answer':result.get('answer'),'score':grade.get('score'),
                         'unsupported_claim':grade.get('unsupported_claim'),'grade_reason':grade.get('reason'),
                         'risk':check.get('risk'),'check_source':check.get('source'),'latency_seconds':result.get('latency_seconds'),
                         'cost_usd':result.get('cost_usd'),'cost_source':result.get('cost_source'),
                         'input_tokens':result.get('input_tokens'),'output_tokens':result.get('output_tokens'),
                         'finish_reason':result.get('finish_reason'),'error':result.get('error')})
        for f in (cp/'answers').glob('*/result.json'):
            result=read(f);ledger.append({'case_id':c.case_id,'model_key':result.get('key'),'stage':'answer', 'cost_usd':result.get('cost_usd'),'cost_source':result.get('cost_source'),'usage':{k:result.get(k) for k in ('input_tokens','output_tokens','total_tokens')}})
        for f in (cp/'answers').glob('*/grade.json'):
            g=read(f);ledger.append({'case_id':c.case_id,'model_key':f.parent.name,'stage':'judge','cost_usd':g.get('cost_usd'),'cost_source':g.get('cost_source'),'usage':g.get('usage')})
    def write_csv(path,data):
        if not data: return
        with path.open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)
    write_csv(run/'comparison.csv',rows)
    (run/'comparison.jsonl').write_text(''.join(encode(r).replace('\n','')+'\n' for r in rows))
    save(run/'cost_ledger.json',ledger)
    groups=[]
    for subset in ('all','overlap_not_detected','jev_only'):
        sub=[r for r in rows if subset=='all' or (subset=='overlap_not_detected' and not r['prior_image_overlap']) or (subset=='jev_only' and r['classification_source']=='jev')]
        for dimension in ('overall','query_type','difficulty','track','location_id'):
            for group in sorted({r.get(dimension,'all') for r in sub}):
                for strategy in protocol['strategies']:
                    selected=[r for r in sub if r['strategy']==strategy and r.get(dimension,'all')==group]
                    if not selected: continue
                    graded=[r for r in selected if r['score'] is not None]
                    valid=[r for r in graded if r['status'] in ('ok','declined')]
                    groups.append({'subset':subset,'dimension':dimension,'group':group,'strategy':strategy,'n':len(selected),'n_graded':len(graded),
                                   'strict_accuracy_all_cases':sum(r['score']==4 and r['status'] in ('ok','declined') for r in graded)/len(selected),
                                   'score_ge_3_all_cases':sum(r['score']>=3 and r['status'] in ('ok','declined') for r in graded)/len(selected),
                                   'mean_score_graded':sum(r['score'] for r in graded)/len(graded) if graded else None,
                                   'errors':sum(r['status']=='error' for r in selected),'truncated':sum(r['status']=='truncated' for r in selected),
                                   'unsupported_claims_graded':sum(bool(r['unsupported_claim']) for r in graded),
                                   'mean_vlm_latency_seconds':sum(r['latency_seconds'] or 0 for r in selected)/len(selected),
                                   'attributed_vlm_cost_usd':sum(r['cost_usd'] or 0 for r in selected),
                                   'model_choices':dict(Counter(r['model_key'] or 'decline' for r in selected))})
    save(run/'summary.json',groups);write_csv(run/'summary.csv',groups)
    known=sum(x['cost_usd'] or 0 for x in ledger)
    status={'updated_at':utc(),'strategy_rows':len(rows),'expected_strategy_rows':len(cases)*3,'graded_strategy_rows':sum(r['score'] is not None for r in rows),
            'unique_answer_records':sum(x['stage']=='answer' for x in ledger),'known_vlm_and_judge_cost_usd':known,
            'cost_limitation':'TypeSafe classifier/checker costs and unreported retry costs are not included; complete SDK replies preserve any exposed usage.',
            'case_count':len(cases),'classifier_sources':dict(Counter(read(f)['classification']['source'] for f in (run/'cases').glob('*/classification.json')))}
    save(run/'status.json',status)
    lines=['# MapEvolve frozen benchmark results','',f'Strategy rows: {len(rows)}; graded: {status["graded_strategy_rows"]}. Known VLM + judge cost: ${known:.4f}. TypeSafe costs may be additional.','',
           'Exploratory results against AI-drafted references; human review pending. GPT-5.1 grades all answers, including its own. 34 cases overlap earlier images.','',
           '| Strategy | Cases | Graded | Strict accuracy | Mean score (0–4) | Errors | Truncated | Attributed VLM cost | Mean VLM latency |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for g in groups:
        if g['subset']=='all' and g['dimension']=='overall':
            mean=f'{g["mean_score_graded"]:.3f}' if g['mean_score_graded'] is not None else 'pending'
            lines.append(f'| {g["strategy"]} | {g["n"]} | {g["n_graded"]} | {g["strict_accuracy_all_cases"]:.1%} | {mean} | {g["errors"]} | {g["truncated"]} | ${g["attributed_vlm_cost_usd"]:.4f} | {g["mean_vlm_latency_seconds"]:.2f}s |')
    lines += ['', 'Accuracy uses all completed strategy rows as the denominator; API errors, missing grades and truncations cannot count as strictly correct. Partial-run results are provisional.', '',
              'See comparison.csv for individual answers and scores; summary.csv/json for type, difficulty, track, location, Jev-only and overlap-not-detected breakdowns. Costs attributed to strategies reuse shared calls and must not be added to estimate actual spend. cost_ledger.json lists unique answer and judge records.', '',
              'Full requests (including encoded images), complete SDK replies, exceptions, prompt text, all classification distributions, action/routing candidates, answer checks, grading rationale and timings are stored under cases/. Credentials are redacted; SDK-hidden retries and internal reasoning are not exposed. No benchmark outputs are ingested into router training.']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return status


def main():
    load_dotenv(ROOT/'.env')
    SECRETS.extend(os.environ.get(k,'') for k in ('TYPESAFE_API_KEY','OPENAI_API_KEY','OPENROUTER_API_KEY'))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='mapevolve-v1-20261115')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-known-cost',type=float,default=10.0)
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--openai-via-openrouter',action='store_true',
                        help='Record an explicit backend amendment: GPT-5.1 via OpenRouter for answers and grading')
    parser.add_argument('--resume-with-updated-runner',action='store_true',
                        help='Archive the new runner version and record its hash as a transport/checkpoint amendment')
    parser.add_argument('--retry-errors',action='store_true',
                        help='Retry saved API errors, preserving their previous result files under history/')
    args=parser.parse_args()
    if Path(args.run_id).name != args.run_id: raise ValueError('run-id must be a directory name')
    cfg,registry,cases,protocol=load_frozen()
    if args.openai_via_openrouter:
        spec=registry.get('OPENAI_FRONTIER_VISION')
        spec['backend']='openrouter'
        spec['model_id']='openai/gpt-5.1'
        spec['chat_kwargs']={'reasoning':{'effort':'none'}}
    policy=RoutingPolicy(registry,cfg,use_quality_model=False)
    run=ROOT/'runs/benchmarks'/args.run_id;run.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:
        print(encode(summarize(run,cases,protocol)));return
    needed=('TYPESAFE_API_KEY','OPENROUTER_API_KEY') if args.openai_via_openrouter else ('TYPESAFE_API_KEY','OPENAI_API_KEY','OPENROUTER_API_KEY')
    for k in needed:
        if not os.environ.get(k): raise ValueError(k+' is missing')
    if not (run/'run.json').exists():
        shutil.copytree(FREEZE,run/'frozen_inputs',dirs_exist_ok=True)
        shutil.copyfile(__file__,run/'runner.py')
        save(run/'effective_registry.json',registry.data)
        save(run/'run.json',{'started_at':utc(),'run_id':args.run_id,'protocol':protocol,'judge_model':'openai/gpt-5.1' if args.openai_via_openrouter else 'gpt-5.1','judge_backend':'openrouter' if args.openai_via_openrouter else 'openai',
                             'backend_amendment':'Direct OpenAI smoke test returned no credits; GPT-5.1 answer/grader moved to OpenRouter, same prompts and frozen prior scores.' if args.openai_via_openrouter else None,
                             'openai_via_openrouter':args.openai_via_openrouter,
                             'max_known_cost_usd':args.max_known_cost,'workers':args.workers,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'cost_limit_scope':'Known answer and judge costs only; TypeSafe billing not exposed by this code.'})
    elif read(run/'run.json')['runner_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        if not args.resume_with_updated_runner:
            raise ValueError('Runner changed since run start; use a new run-id or explicitly record an amendment')
        meta=read(run/'run.json')
        amendments=meta.setdefault('runner_amendments',[])
        current_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        amendments.append({'timestamp':utc(),'previous_sha256':meta['runner_sha256'],'new_sha256':current_hash,
                           'reason':'Transport pacing at 15 GPT-5.1 calls/minute and explicit retry of saved API errors; prompts/models/routing unchanged.'})
        shutil.copyfile(__file__,run/f'runner_amendment_{len(amendments)}.py')
        meta['runner_sha256']=current_hash
        save(run/'run.json',meta)
    elif read(run/'run.json').get('openai_via_openrouter',False) != args.openai_via_openrouter:
        raise ValueError('Backend differs from run start')
    class RedactLog(logging.Filter):
        def filter(self,record):
            text=record.getMessage()
            for secret in SECRETS:
                if secret: text=text.replace(secret,'[REDACTED]')
            record.msg=text;record.args=()
            return True
    log_handler=logging.FileHandler(run/'sdk.log');log_handler.addFilter(RedactLog());logging.getLogger().addHandler(log_handler)
    work=cases[:args.limit] if args.limit else cases
    def cost_guard():
        known=0
        for f in (run/'cases').glob('*/answers/*/result.json'): known += read(f).get('cost_usd') or 0
        for f in (run/'cases').glob('*/answers/*/grade.json'): known += read(f).get('cost_usd') or 0
        if known >= args.max_known_cost: raise RuntimeError('Known cost guard reached')
    def execute(case):
        cp=run/'cases'/case.case_id;cp.mkdir(parents=True,exist_ok=True)
        inp=routing_inputs(case);cost_guard()
        if (cp/'classification.json').exists():
            classification=QueryClassification.from_dict(read(cp/'classification.json')['classification'])
        else:
            clf=JevQueryClassifier(model=cfg['jev']['model'],timeout=cfg['jev']['timeout_seconds'],image_aware=True)
            clf._decision_model=Recorder(typesafe_classifier(model=cfg['jev']['model'],timeout=cfg['jev']['timeout_seconds']),cp/'classifier_sdk')
            start=time.perf_counter();classification=clf.classify(inp['question'],inp['images'],image_aware=True,evidence=inp['supplied_evidence'])
            save(cp/'classification.json',{'classification':classification.to_dict(),'wall_seconds':time.perf_counter()-start})
        action=decide_action(classification,cfg,inp['supplied_evidence'])
        decision=policy.route(classification,len(inp['images']),protocol['adaptive_mode'],protocol['budget_usd_per_query'],True,protocol['allowed_model_keys'])
        save(cp/'routing.json',{'action':action.to_dict(),'routing':decision.to_dict()})
        selected={'cheapest_eligible':protocol['cheapest_baseline'],'jev_adaptive':None if action.action=='decline' else decision.selected_key,'strongest':protocol['strongest_baseline']}
        save(cp/'strategies.json',selected)
        for key in dict.fromkeys(selected.values()):
            folder=cp/'answers'/(key or 'adaptive_decline');folder.mkdir(parents=True,exist_ok=True)
            if args.retry_errors and (folder/'result.json').exists() and read(folder/'result.json')['status']=='error':
                history=folder/'history'/str(time.time_ns());history.mkdir(parents=True)
                for filename in ('result.json','check.json','grade.json'):
                    if (folder/filename).exists(): shutil.move(str(folder/filename),history/filename)
            if not (folder/'result.json').exists():
                cost_guard()
                if key is None:
                    result={'key':None,'status':'declined','answer':decline_message(classification),'cost_usd':0,'cost_source':'no_call','latency_seconds':0}
                else:
                    def factory(model,generation):
                        save(folder/'model_config.json',{'model':model,'generation':generation})
                        return Recorder(default_chat_factory(model,generation),folder/'vlm_sdk')
                    executor=VLMExecutor(registry,cfg['generation'],chat_factory=factory)
                    result=executor.run(key,inp['question'],inp['images'],task=classification.query_type.label,evidence=inp['supplied_evidence'],note=action.prompt_note).to_dict()
                save(folder/'result.json',result)
            result=read(folder/'result.json')
            if result.get('answer') and not (folder/'check.json').exists():
                checker=AnswerChecker(decision_model=Recorder(typesafe_classifier(model=cfg['jev']['model'],timeout=cfg['jev']['timeout_seconds']),folder/'checker_sdk'))
                start=time.perf_counter();check=checker.check(result['answer'],inp['question'],inp['images'],classification,inp['supplied_evidence']).to_dict()
                save(folder/'check.json',{**check,'wall_seconds':time.perf_counter()-start})
            if result.get('answer') and not (folder/'grade.json').exists():
                cost_guard()
                client=Recorder(default_chat_factory(registry.get('OPENAI_FRONTIER_VISION'),cfg['generation']),folder/'judge_sdk')
                reply=client.invoke([HumanMessage(content=judge_prompt(case,result['answer']))])
                grade=normalize_grade(parse_verdict(reply.content))
                usage=reply.usage_metadata or {};meta=reply.response_metadata or {}
                price=registry.get('OPENAI_FRONTIER_VISION')['pricing']
                estimated=(usage.get('input_tokens',0)*price['input_usd_per_mtok']+usage.get('output_tokens',0)*price['output_usd_per_mtok'])/1e6
                grade.update(judge_model=registry.get('OPENAI_FRONTIER_VISION')['model_id'],usage=usage,cost_usd=meta.get('cost',estimated),cost_source='provider_reported' if meta.get('cost') is not None else 'estimated_from_registry_pricing')
                save(folder/'grade.json',grade)
        save(cp/'complete.json',{'finished_at':utc(),'classification_source':classification.source})
        return case.case_id,classification.source,selected
    failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(execute,c):c for c in work}
        for future in as_completed(futures):
            c=futures[future]
            try:
                cid,source,selected=future.result();print(f'COMPLETE {cid} {source} {selected}',flush=True)
            except Exception as exc:
                failures.append({'case_id':c.case_id,'exception_type':type(exc).__name__,'error':str(exc)})
                save(run/'cases'/c.case_id/'case_error.json',failures[-1]);print(f'FAILED {c.case_id} {type(exc).__name__}',flush=True)
            with IO_LOCK: summarize(run,cases,protocol)
    save(run/'last_execution.json',{'finished_at':utc(),'processed_cases':len(work),'failures':failures})
    print(encode(summarize(run,cases,protocol)),flush=True)
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
