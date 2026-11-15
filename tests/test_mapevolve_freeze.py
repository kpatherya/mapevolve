import hashlib
import json
import pytest
from scripts import verify_mapevolve_freeze as frozen


def test_frozen_benchmark_is_loadable():
    cfg, registry, cases, protocol = frozen.load_frozen()
    assert len(cases) == 100
    assert len(protocol['prior_image_overlap_case_ids']) == 34
    assert protocol['cheapest_baseline'] == 'GEMMA3_4B'
    assert protocol['strongest_baseline'] == 'OPENAI_FRONTIER_VISION'
    assert registry.get('NEMOTRON_VL')['pricing']['output_usd_per_mtok'] == 0.5


def test_freeze_rejects_modified_inputs(tmp_path, monkeypatch):
    p = tmp_path/'input.txt'
    p.write_text('original')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'lock.json').write_text(json.dumps({'sha256':{'input.txt':digest}}))
    p.write_text('changed')
    monkeypatch.setattr(frozen,'FREEZE',tmp_path)
    monkeypatch.setattr(frozen,'ROOT',tmp_path)
    with pytest.raises(ValueError, match='Frozen benchmark inputs changed: input.txt'):
        frozen.load_frozen()
