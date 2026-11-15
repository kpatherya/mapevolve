"""Export full per-strategy diagnostics and audit a completed frozen run.

No network access or model calls. Usage:
python -m scripts.analyze_frozen_benchmark runs/benchmarks/<run-id>
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
from jev_router.dataset import Case, ROOT


def load(path):return json.loads(path.read_text())


def flatten(prefix,value,out):
    if isinstance(value,dict):
        for k,v in value.items():flatten(f'{prefix}.{k}' if prefix else k,v,out)
    elif isinstance(value,list):out[prefix]=json.dumps(value,ensure_ascii=False)
    else:out[prefix]=value


def export(run):
    frozen=load(run/'frozen_inputs/cases.json');cases={c['case_id']:Case.from_dict(c) for c in frozen['cases']}
    protocol=load(run/'run.json')['protocol'];rows=[];usage=[];audit=[];metadata=[];classifier_rows=[];routing_rows=[]
    for cid,c in cases.items():
        folder=run/'cases'/cid
        if not (folder/'strategies.json').exists():continue
        cls=load(folder/'classification.json');routing=load(folder/'routing.json');choices=load(folder/'strategies.json')
        classifier_rows.append({'case_id':cid,**cls})
        routing_rows.append({'case_id':cid,**routing,'strategies':choices})
        for strategy,key in choices.items():
            af=folder/'answers'/(key or 'adaptive_decline')
            if not (af/'result.json').exists():continue
            result=load(af/'result.json');grade=load(af/'grade.json') if (af/'grade.json').exists() else {}
            check=load(af/'check.json') if (af/'check.json').exists() else {}
            sdk_replies=sorted((af/'vlm_sdk').glob('response_*.json'),key=lambda p:int(p.stem.split('_')[-1]))
            api_seconds=load(sdk_replies[-1])['elapsed_seconds'] if sdk_replies else (0 if result['status']=='declined' else None)
            row={'case_id':cid,'strategy':strategy,'model_key':key,'query_type_label':c.query_type,'difficulty_label':c.difficulty,'track':c.tags[0],'location_id':c.location_id,
                 'question':c.question,'image_paths':json.dumps(c.images),'supplied_evidence':c.supplied_evidence,'reference_answer':c.reference['answer'],
                 'reference_required_points':json.dumps(c.reference['required_points']),'prior_image_overlap':cid in protocol['prior_image_overlap_case_ids'],
                 'sdk_api_seconds_excluding_pacing':api_seconds,
                 'client_wait_and_setup_seconds':max(0,(result.get('latency_seconds') or 0)-api_seconds) if api_seconds is not None else None}
            for label,value in [('classification',cls),('action',routing['action']),('routing',routing['routing']),('answer',result),('checker',check),('grade',grade)]:flatten(label,value,row)
            rows.append(row)
            if 'capture_metadata' in c.tags:
                expected=re.search(r'\b(?:19|20)\d{2}(?:-\d{2}-\d{2})?\b',c.reference['answer']).group()
                text=result.get('answer') or ''
                matches=expected in text
                metadata.append({'case_id':cid,'strategy':strategy,'model_key':key,'expected_recorded_date':expected,'expected_date_literal_present':matches,
                                 'other_years_in_answer':json.dumps(sorted(set(re.findall(r'\b(?:19|20)\d{2}\b',text))-{expected[:4]})),
                                 'judge_score':grade.get('score'),'note':'Literal match is a cross-check, not a replacement for rubric scoring.'})
                if not matches or (grade.get('score',0)<4 and matches):
                    audit.append({'case_id':cid,'strategy':strategy,'model_key':key,'reason':'Metadata literal/reference vs grade requires review','answer':text,'reference':c.reference['answer'],'grade':grade})
            if grade.get('score') in (2,3) or grade.get('unsupported_claim') or result['status']!='ok':
                audit.append({'case_id':cid,'strategy':strategy,'model_key':key,'reason':'Partial/unsupported/failed answer','answer':result.get('answer'),'reference':c.reference['answer'],'grade':grade})
        for f in folder.rglob('response_*.json'):
            reply=load(f);body=reply.get('response',{});tokens=body.get('usage') or body.get('usage_metadata') or {}
            stage='classifier' if 'classifier_sdk' in str(f) else 'checker' if 'checker_sdk' in str(f) else 'judge' if 'judge_sdk' in str(f) else 'vlm'
            meta=body.get('response_metadata') or {}
            usage.append({'case_id':cid,'stage':stage,'artifact':str(f.relative_to(run)),'elapsed_seconds':reply.get('elapsed_seconds'),
                          'input_tokens':tokens.get('input_tokens'),'output_tokens':tokens.get('output_tokens'),'total_tokens':tokens.get('total_tokens'),
                          'provider_reported_cost':meta.get('cost'),'returned_model':meta.get('model_name') or body.get('model'),
                          'request_id':meta.get('id') or body.get('request_id'),'finish_reason':meta.get('finish_reason')})
    def csvfile(name,data):
        if data:
            with (run/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)
    csvfile('full_diagnostics.csv',rows);csvfile('all_sdk_usage.csv',usage);csvfile('metadata_crosschecks.csv',metadata)
    flat_classifications=[]
    for record in classifier_rows:
        row={};flatten('',record,row);flat_classifications.append(row)
    csvfile('classifications.csv',flat_classifications)
    (run/'classifications.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in classifier_rows))
    (run/'routing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in routing_rows))
    (run/'adjudication_queue.json').write_text(json.dumps(audit,indent=2)+'\n')
    # Macro-location scores avoid letting many views of one building dominate.
    summaries=load(run/'summary.json');macro=[]
    latency_groups=[]
    for g in summaries:
        subset=g['subset'];dimension=g['dimension']
        selected=[r for r in rows if r['strategy']==g['strategy']
                  and (subset=='all' or (subset=='overlap_not_detected' and not r['prior_image_overlap']) or (subset=='jev_only' and r['classification.classification.source']=='jev'))
                  and (dimension=='overall' or r[{'query_type':'query_type_label','difficulty':'difficulty_label'}.get(dimension,dimension)]==g['group'])]
        # Difficulty uses the manual benchmark label, not Jev's predicted tier.
        api_values=[r['sdk_api_seconds_excluding_pacing'] for r in selected if r['sdk_api_seconds_excluding_pacing'] is not None]
        latency_groups.append({**g,'mean_sdk_api_seconds_excluding_pacing':sum(api_values)/len(api_values) if api_values else None})
    csvfile('summary_with_api_latency.csv',latency_groups)
    for strategy in protocol['strategies']:
        groups=[g for g in summaries if g['subset']=='all' and g['dimension']=='location_id' and g['strategy']==strategy]
        macro.append({'strategy':strategy,'n_locations':len(groups),'macro_location_strict_accuracy':sum(g['strict_accuracy_all_cases'] for g in groups)/len(groups) if groups else None})
    (run/'location_macro_scores.json').write_text(json.dumps(macro,indent=2)+'\n')
    stages={s:{'calls':sum(u['stage']==s for u in usage),'input_tokens':sum(u['input_tokens'] or 0 for u in usage if u['stage']==s),
               'output_tokens':sum(u['output_tokens'] or 0 for u in usage if u['stage']==s),'summed_call_seconds':sum(u['elapsed_seconds'] or 0 for u in usage if u['stage']==s),
               'provider_reported_cost_usd':sum(u['provider_reported_cost'] or 0 for u in usage if u['stage']==s)} for s in ('classifier','checker','vlm','judge')}
    checks={'strategy_rows':len(rows),'all_100_cases_complete':all((run/'cases'/cid/'complete.json').exists() for cid in cases),
            'all_300_strategy_rows_present':len(rows)==300,'all_strategy_rows_graded':all('grade.score' in r for r in rows),
            'classification_sources':dict(Counter(load(f)['classification']['source'] for f in (run/'cases').glob('*/classification.json'))),
            'answer_statuses_by_strategy':{s:dict(Counter(r['answer.status'] for r in rows if r['strategy']==s)) for s in protocol['strategies']},
            'check_sources':dict(Counter(r.get('checker.source') for r in rows)),
            'sdk_usage_by_stage':stages,'adjudication_queue_rows':len(audit),'metadata_literal_matches':sum(r['expected_date_literal_present'] for r in metadata),'metadata_rows':len(metadata),
            'notes':['SDK request counts exclude internal retries not exposed by clients.','TypeSafe dollar costs are unreported, not zero.','Full diagnostics includes repeated data for reused answers; all_sdk_usage counts unique exposed calls.']}
    (run/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);args=p.parse_args();export(args.run)
