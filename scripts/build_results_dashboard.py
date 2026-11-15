"""Build a portable, offline HTML explorer of a completed benchmark.

python -m scripts.build_results_dashboard
No API calls. Source results and frozen inputs remain unchanged.
"""
import argparse
import base64
import io
import json
from pathlib import Path
from PIL import Image, ImageOps
from jev_router.dataset import ROOT


def load(p): return json.loads(p.read_text())


def build(run, output):
    frozen = load(run/'frozen_inputs/cases.json')
    protocol = load(run/'run.json')['protocol']
    registry = {m['key']:m for m in load(run/'effective_registry.json')['models']}
    images = {};cases = []
    for c in frozen['cases']:
        folder = run/'cases'/c['case_id']
        image = c['images'][0]
        if image not in images:
            with Image.open(ROOT/image) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((1000,1000)); buffer=io.BytesIO();im.save(buffer,'JPEG',quality=82)
                images[image]='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()
        classification=load(folder/'classification.json')
        routing=load(folder/'routing.json')
        answers={}
        for strategy,key in load(folder/'strategies.json').items():
            af=folder/'answers'/(key or 'adaptive_decline')
            replies=sorted((af/'vlm_sdk').glob('response_*.json'),key=lambda p:int(p.stem.split('_')[-1]))
            sdk=load(replies[-1]) if replies else {}
            label='GPT-5.1 · OpenRouter' if key=='OPENAI_FRONTIER_VISION' and registry[key]['backend']=='openrouter' else registry.get(key,{}).get('display_name',key or 'Declined')
            answers[strategy]={'model':key,'model_label':label,
                              'result':load(af/'result.json'),'grade':load(af/'grade.json'),
                              'check':load(af/'check.json') if (af/'check.json').exists() else {},
                              'api_seconds':sdk.get('elapsed_seconds',0 if key is None else None),
                              'response_metadata':sdk.get('response',{}).get('response_metadata',{})}
        cases.append({**c,'image':image,'overlap':c['case_id'] in protocol['prior_image_overlap_case_ids'],
                      'classification':classification,'routing':routing,'answers':answers})
    data={'cases':cases,'images':images,'status':load(run/'status.json'),'verification':load(run/'verification.json'),
          'run':load(run/'run.json'),'completion':load(run/'completion.json'),'macro':load(run/'location_macro_scores.json')}
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    template=(ROOT/'reports/dashboard-template.html').read_text()
    assert template.count('__BENCHMARK_DATA__')==1
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(template.replace('__BENCHMARK_DATA__',encoded))
    print(f'Built {output}: {len(cases)} cases, {len(images)} embedded images; {output.stat().st_size/1024**2:.1f} MiB')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/benchmarks/mapevolve-v1-openrouter-20261115')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/mapevolve-results.html')
    args=parser.parse_args();build(args.run,args.output)
